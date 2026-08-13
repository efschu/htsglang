"""#605 fill side: the recorder must not misfile a rank or hide a residue.

WHY THIS SUITE EXISTS. #602 closed the break side and left the fill side open:
the three cards of the reference rig finish a boot at 4593 / 2158 / 2854 MiB
free against a 1024 MiB corridor, and no report in the tree could name what
holds the difference. Reading the marks of a real ship boot
(1353495-1786609875, 2026-08-13) showed the instrument itself was the reason,
in two ways, and each one is pinned below by a test that FAILS against the
pre-fix recorder.

DEFECT A -- rank collision under pipeline parallelism. Marks are filed under
``flight_marks_rank{rank}.jsonl`` where ``rank`` is the TP rank. The ship
config runs ``--tp-size 1 --pp-size 3``, so the TP rank is 0 in ALL THREE
processes: every rank writes its early marks into rank 0's file, and a reader
that groups by rank builds one timeline out of three different cards. In the
ship boot that put a 20480 MiB card's CUDA context into the 32607 MiB card's
column. The pid is stamped on every mark by the process that took it and is
never ambiguous, so grouping is by pid.

DEFECT B -- the residue is floored to zero exactly where it is interesting.
``non_torch_bytes`` is written as ``max(0, nvml_self - torch_reserved)``. On
the ship config torch reports 7162 / 5514 / 4824 MiB MORE reserved than NVML
says the process physically holds, so the subtraction goes far negative and the
field reads 0 -- which a reader takes to mean "no CUDA context, no NCCL, no JIT
workspace on this card", while the context alone is 886 MiB. A floor that was
meant to absorb sub-MiB quantisation swallows gigabytes and reports a false
zero. The overshoot is a measurement in its own right and gets its own field.
"""

import json
import os
import sys
import tempfile
import unittest

from sglang.srt.mem_ledger import flight_recorder
from sglang.srt.mem_ledger.flight_recorder import MIB, mark, read_marks

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts"
    ),
)


def _fake_views(monkey_self, reserved_bytes, nvml_self_bytes, card_uuid):
    """Pin the two view functions so a mark can be taken with no GPU present."""

    def torch_view(device_index=None):
        return {
            "cuda_initialized": True,
            "allocated_bytes": reserved_bytes,
            "allocated_peak_bytes": reserved_bytes,
            "reserved_bytes": reserved_bytes,
            "reserved_peak_bytes": reserved_bytes,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }

    def nvml_view():
        return {
            "card_uuid": card_uuid,
            "nvml_total_bytes": 32607 * MIB,
            "nvml_free_bytes": 32607 * MIB - nvml_self_bytes,
            "nvml_used_bytes": nvml_self_bytes,
            "nvml_carve_out_bytes": 518 * MIB,
            "nvml_self_bytes": nvml_self_bytes,
            "nvml_processes": {str(os.getpid()): nvml_self_bytes},
        }

    monkey_self._saved = (flight_recorder._torch_view, flight_recorder._nvml_view)
    flight_recorder._torch_view = torch_view
    flight_recorder._nvml_view = nvml_view


class TestRankCollisionUnderPP(unittest.TestCase):
    """DEFECT A: three processes, one TP rank, three cards."""

    def test_marks_of_three_pids_do_not_merge_into_one_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            # Three processes that all believe they are TP rank 0, each on its
            # own card -- the ship config's PP=3 / TP=1 shape, written by hand
            # because the collision only appears with three real processes.
            cards = {
                4001: "GPU-aaaa",
                4002: "GPU-bbbb",
                4003: "GPU-cccc",
            }
            path = os.path.join(directory, "flight_marks_rank0.jsonl")
            with open(path, "w") as handle:
                for pid, uuid in cards.items():
                    for phase in ("process_start", "pre_weight_load"):
                        handle.write(
                            json.dumps(
                                {
                                    "phase": phase,
                                    "rank": 0,
                                    "boot_id": "b1",
                                    "pid": pid,
                                    "wall": 1.0,
                                    "monotonic": 1.0,
                                    "card_uuid": uuid,
                                    "nvml_self_bytes": 0,
                                    "reserved_bytes": 0,
                                }
                            )
                            + "\n"
                        )

            grouped = read_marks(directory)
            # The contract: one group per WRITER, never one group per rank
            # field. Three processes wrote here; three groups must come back.
            self.assertEqual(
                len(grouped),
                3,
                f"three pids wrote marks, reader returned {len(grouped)} group(s); "
                "grouping by rank merges three cards into one timeline",
            )
            for key, group in grouped.items():
                seen = {m["card_uuid"] for m in group}
                self.assertEqual(
                    len(seen),
                    1,
                    f"group {key} mixes cards {sorted(seen)}; a group must be one card",
                )


class TestResidueIsNotFlooredToZero(unittest.TestCase):
    """DEFECT B: reserved above resident must be reported, not clamped away."""

    def tearDown(self):
        saved = getattr(self, "_saved", None)
        if saved:
            flight_recorder._torch_view, flight_recorder._nvml_view = saved

    def test_unbacked_reservation_is_named_when_reserved_exceeds_resident(self):
        with tempfile.TemporaryDirectory() as directory:
            # The ship boot's rank 0: torch says 34470 MiB reserved, NVML says
            # the process holds 27308 MiB. 7162 MiB of reservation carries no
            # physical backing.
            _fake_views(self, 34470 * MIB, 27308 * MIB, "GPU-aaaa")
            record = mark("first_forward", rank=0, directory=directory)

        self.assertIsNotNone(record)
        self.assertIn(
            "unbacked_reservation_bytes",
            record,
            "reserved above resident is a measurement and needs its own field",
        )
        self.assertEqual(record["unbacked_reservation_bytes"], 7162 * MIB)
        self.assertFalse(
            record["non_torch_measurable"],
            "with reserved above resident the non-torch residue cannot be "
            "derived by subtraction, and the mark must say so",
        )

    def test_ordinary_boot_still_measures_the_non_torch_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            # The same rank earlier in the same boot: 888 MiB resident, 2 MiB
            # of it torch's. The 886 MiB context is the residue, and this is
            # the case the field was built for.
            _fake_views(self, 2 * MIB, 888 * MIB, "GPU-aaaa")
            record = mark("pre_weight_load", rank=0, directory=directory)

        self.assertEqual(record["non_torch_bytes"], 886 * MIB)
        self.assertTrue(record["non_torch_measurable"])
        self.assertEqual(record["unbacked_reservation_bytes"], 0)


class TestFillSideReportCatchesAnUnbookedAllocation(unittest.TestCase):
    """Stage 2 falsifier: an unbooked post must not vanish into a tidy table.

    A deliberate allocation is planted that no phase mark brackets -- the
    module that made it took no mark, which is exactly how a real unbooked
    consumer behaves. The report may not absorb it: either it is named, or the
    residual line grows by its size. Silence is the failure being pinned.
    """

    def _marks(self, plant_bytes):
        base = dict(
            boot_id="b1",
            pid=4001,
            card_uuid="GPU-aaaa",
            nvml_total_bytes=32607 * MIB,
            nvml_carve_out_bytes=518 * MIB,
            rank=0,
        )
        resident = [0, 888 * MIB, 15592 * MIB]
        phases = ["process_start", "pre_weight_load", "weights_loaded"]
        marks = []
        for index, (phase, self_bytes) in enumerate(zip(phases, resident)):
            marks.append(
                dict(
                    base,
                    phase=phase,
                    monotonic=float(index),
                    wall=float(index),
                    nvml_self_bytes=self_bytes,
                    reserved_bytes=self_bytes,
                    # The plant sits on the CARD but outside this pid: a side
                    # module in another process, the shape of the tokenizer's
                    # second context and of a lane pool outside the rank.
                    nvml_used_bytes=self_bytes + 518 * MIB + plant_bytes,
                    nvml_free_bytes=(
                        32607 * MIB - self_bytes - 518 * MIB - plant_bytes
                    ),
                )
            )
        return marks

    def test_planted_300_mib_moves_the_residual_line_by_300_mib(self):
        from vram_ledger.fill_side_report import report

        clean, ok_clean = report(self._marks(0))
        planted, ok_planted = report(self._marks(300 * MIB))
        self.assertTrue(ok_clean, f"clean report must close:\n{clean}")
        self.assertTrue(ok_planted, f"planted report must still close:\n{planted}")

        def residual(text):
            for line in text.splitlines():
                if "foreign_or_unattributed" in line:
                    return int(line.split()[1])
            raise AssertionError(f"no residual line in report:\n{text}")

        self.assertEqual(
            residual(planted) - residual(clean),
            300,
            "a 300 MiB allocation nobody booked must appear in the residual, "
            f"not be absorbed:\nCLEAN\n{clean}\nPLANTED\n{planted}",
        )


if __name__ == "__main__":
    unittest.main()
