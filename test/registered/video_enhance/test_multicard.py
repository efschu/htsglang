"""Multi-card chunk execution: the seam, the arithmetic, and the ordering.

Everything here runs on CPU with a fake chunk runner. That is deliberate: the
three properties a multi-card run has to have are all decidable without a
card, and none of them is decidable from a throughput number.

*   The seam frame is interpolated once and encoded once. Encoding it on both
    sides duplicates a frame per boundary; on neither side drops one. Both
    are invisible in the output file and both desynchronise the audio.
*   Whatever the chunking, the output frame count equals the single-card
    ``expected_frame_count``. The muxer retimes against that number, so a
    chunking that changes it puts A/V out by construction.
*   Chunks reach the sink in timeline order even though the cards finish out
    of order, and a chunk that has finished early does not get to occupy an
    unbounded amount of spool while an earlier one is still streaming.
"""

import asyncio
import unittest
from pathlib import Path

from sglang.srt.video_enhance.chain import ChainRequest, build_chain
from sglang.srt.video_enhance.frame_math import MIB, Resolution
from sglang.srt.video_enhance.multicard import (
    ChunkResult,
    ChunkRunner,
    ChunkSpec,
    MultiCardError,
    MultiCardExecutor,
    SubprocessChunkRunner,
    chunk_specs_from_plan,
    total_output_frames,
    verify_chunk_arithmetic,
)
from sglang.srt.video_enhance.mux import expected_frame_count
from sglang.srt.video_enhance.shard_plan import (
    CardAvailability,
    RateTable,
    ReservationInputs,
    StageKind,
    StageRate,
    capacity_weighted_plan,
    static_single_card_plan,
    vsgan_style_modulo_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


SOURCE = Resolution(1920, 1080)
TARGET = Resolution(1920, 1080)


def rife_only_chain(multiplier: int = 2):
    """The 4K-style configuration: interpolation only, no SR and no resize."""
    return build_chain(
        ChainRequest(
            source=SOURCE,
            target=TARGET,
            fps_multiplier=multiplier,
            enable_sr=False,
            enable_resize=False,
            streams_in_flight=1,
        )
    )


def rate_table(cards, ms):
    """A P1 table covering the rife-only chain on the given cards."""
    rows = []
    for card in cards:
        for kind in (
            StageKind.DECODE,
            StageKind.COLOR_TO_RGB,
            StageKind.RIFE,
            StageKind.COLOR_TO_YUV,
            StageKind.ENCODE,
        ):
            rows.append(StageRate(kind, card, SOURCE, ms[card]))
    return RateTable(rows)


def availabilities(cards, budget_mib=8192):
    return [CardAvailability(card=c, reserved_bytes=budget_mib * MIB) for c in cards]


RESERVATION = ReservationInputs(rife_measured_bytes_per_pair=1185 * MIB // 1)


class ChunkArithmeticTest(CustomTestCase):
    """The frame counts, which are the whole correctness argument."""

    def test_a_single_chunk_is_the_single_card_answer(self):
        chunk = ChunkSpec(
            index=0, card="1", start=0, stop=100, pulls_successor=False, multiplier=2
        )
        self.assertEqual(chunk.output_frames, expected_frame_count(100, 2))
        self.assertEqual(chunk.output_frames, 199)

    def test_a_chunk_with_a_successor_encodes_owned_times_multiplier(self):
        """n owned frames, n pairs including the seam, trailing frame withheld."""
        chunk = ChunkSpec(
            index=0, card="1", start=0, stop=40, pulls_successor=True, multiplier=2
        )
        self.assertEqual(chunk.pulled_frames, 41)
        self.assertEqual(chunk.output_frames, 80)

    def test_two_chunks_add_up_to_the_single_card_count(self):
        chunks = [
            ChunkSpec(0, "1", 0, 40, True, 2),
            ChunkSpec(1, "0", 40, 100, False, 2),
        ]
        self.assertEqual(total_output_frames(chunks), expected_frame_count(100, 2))
        verify_chunk_arithmetic(chunks, 100, 2)

    def test_the_count_holds_for_any_split_and_any_multiplier(self):
        for total in (7, 48, 100, 997):
            for multiplier in (1, 2, 3, 4):
                for cut in (1, total // 3, total // 2, total - 1):
                    if not 0 < cut < total:
                        continue
                    chunks = [
                        ChunkSpec(0, "a", 0, cut, multiplier > 1, multiplier),
                        ChunkSpec(1, "b", cut, total, False, multiplier),
                    ]
                    self.assertEqual(
                        total_output_frames(chunks),
                        expected_frame_count(total, multiplier),
                        f"total={total} m={multiplier} cut={cut}",
                    )

    def test_three_chunks_also_add_up(self):
        chunks = [
            ChunkSpec(0, "a", 0, 30, True, 3),
            ChunkSpec(1, "b", 30, 55, True, 3),
            ChunkSpec(2, "c", 55, 90, False, 3),
        ]
        verify_chunk_arithmetic(chunks, 90, 3)

    def test_multiplier_one_needs_no_seam_frame(self):
        """Without interpolation nothing reads across frames, so no overlap."""
        chunks = [
            ChunkSpec(0, "a", 0, 50, False, 1),
            ChunkSpec(1, "b", 50, 100, False, 1),
        ]
        verify_chunk_arithmetic(chunks, 100, 1)
        self.assertEqual(total_output_frames(chunks), 100)

    def test_forgetting_the_seam_frame_is_caught(self):
        """The failure mode this check exists for: one frame short per seam."""
        chunks = [
            ChunkSpec(0, "a", 0, 40, False, 2),
            ChunkSpec(1, "b", 40, 100, False, 2),
        ]
        with self.assertRaises(MultiCardError) as ctx:
            verify_chunk_arithmetic(chunks, 100, 2)
        self.assertIn("198", str(ctx.exception))

    def test_a_gap_in_the_cover_is_caught(self):
        chunks = [
            ChunkSpec(0, "a", 0, 40, True, 2),
            ChunkSpec(1, "b", 41, 100, False, 2),
        ]
        with self.assertRaises(MultiCardError):
            verify_chunk_arithmetic(chunks, 100, 2)

    def test_an_empty_chunk_is_refused_at_construction(self):
        with self.assertRaises(MultiCardError):
            ChunkSpec(0, "a", 10, 10, False, 2)


class SeamPredicateTest(CustomTestCase):
    """Which frames a chunk encodes, as a pure function."""

    def setUp(self):
        self.chunk = ChunkSpec(0, "a", 10, 20, True, 3)

    def test_owned_originals_are_encoded(self):
        for index in range(10, 20):
            self.assertTrue(self.chunk.encodes(index, 0), index)

    def test_the_seam_original_is_withheld(self):
        self.assertFalse(self.chunk.encodes(20, 0))

    def test_interpolated_frames_at_the_seam_are_kept(self):
        """They are keyed to frame 19, the pair's earlier frame, and are ours."""
        self.assertTrue(self.chunk.encodes(19, 1))
        self.assertTrue(self.chunk.encodes(19, 2))

    def test_the_last_chunk_withholds_nothing(self):
        last = ChunkSpec(1, "b", 20, 30, False, 3)
        self.assertTrue(last.encodes(30, 0))
        self.assertIsNone(last.successor_index)


class PlanToChunksTest(CustomTestCase):
    def setUp(self):
        self.chain = rife_only_chain(2)
        self.cards = ["1", "0", "2"]
        # 5090 roughly twice a 3080 on this chain.
        self.rates = rate_table(self.cards, {"1": 10.0, "0": 20.0, "2": 20.0})

    def test_capacity_weighted_plan_becomes_a_valid_chunking(self):
        plan = capacity_weighted_plan(
            chain=self.chain,
            rates=self.rates,
            cards=availabilities(self.cards),
            total_frames=400,
            reservation=RESERVATION,
        )
        chunks = chunk_specs_from_plan(plan, multiplier=2)
        verify_chunk_arithmetic(chunks, 400, 2)
        self.assertEqual(len(chunks), 3)
        # Every chunk but the last pulls a seam frame.
        self.assertEqual([c.pulls_successor for c in chunks], [True, True, False])

    def test_the_fast_card_gets_the_larger_share(self):
        plan = capacity_weighted_plan(
            chain=self.chain,
            rates=self.rates,
            cards=availabilities(self.cards),
            total_frames=400,
            reservation=RESERVATION,
        )
        chunks = {
            c.card: c.owned_frames for c in chunk_specs_from_plan(plan, multiplier=2)
        }
        self.assertGreater(chunks["1"], chunks["0"])
        self.assertEqual(chunks["0"], chunks["2"])

    def test_single_card_plan_is_one_chunk_with_no_seam(self):
        plan = static_single_card_plan(
            chain=self.chain,
            rates=self.rates,
            cards=availabilities(self.cards),
            total_frames=400,
            reservation=RESERVATION,
        )
        chunks = chunk_specs_from_plan(plan, multiplier=2)
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].pulls_successor)
        verify_chunk_arithmetic(chunks, 400, 2)

    def test_the_modulo_baseline_is_refused_rather_than_run(self):
        """It is a costed baseline. Running it would split every RIFE pair."""
        plan = vsgan_style_modulo_plan(
            chain=self.chain,
            rates=self.rates,
            cards=availabilities(self.cards),
            total_frames=400,
            reservation=RESERVATION,
        )
        with self.assertRaises(MultiCardError) as ctx:
            chunk_specs_from_plan(plan, multiplier=2)
        self.assertIn("contiguous", str(ctx.exception))

    def test_a_chain_without_rife_needs_no_seam_frames(self):
        """A 4K-to-1080p downscale: no stage reads across frames, so no seam."""
        uhd = Resolution(3840, 2160)
        chain = build_chain(
            ChainRequest(
                source=uhd,
                target=SOURCE,
                fps_multiplier=1,
                enable_sr=False,
                enable_resize=True,
                streams_in_flight=1,
            )
        )
        rates = RateTable(
            [
                StageRate(spec.kind, card, spec.in_res, 5.0)
                for card in self.cards
                for spec in chain.stages
            ]
        )
        plan = capacity_weighted_plan(
            chain=chain,
            rates=rates,
            cards=availabilities(self.cards),
            total_frames=300,
            reservation=RESERVATION,
        )
        chunks = chunk_specs_from_plan(plan, multiplier=1)
        self.assertTrue(all(not c.pulls_successor for c in chunks))
        verify_chunk_arithmetic(chunks, 300, 1)


class FakeRunner(ChunkRunner):
    """Writes a recognisable payload per chunk, after a controllable delay."""

    def __init__(self, delays=None, frames_override=None):
        self.delays = delays or {}
        self.frames_override = frames_override or {}
        self.started: list[int] = []
        self.finished: list[int] = []
        self.concurrent = 0
        self.peak_concurrent = 0

    async def run(self, chunk: ChunkSpec, spool: Path) -> ChunkResult:
        self.started.append(chunk.index)
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.delays.get(chunk.index, 0))
            payload = f"<chunk{chunk.index}>".encode() * 4
            spool.parent.mkdir(parents=True, exist_ok=True)
            spool.write_bytes(payload)
            self.finished.append(chunk.index)
            return ChunkResult(
                index=chunk.index,
                card=chunk.card,
                path=spool,
                frames_encoded=self.frames_override.get(
                    chunk.index, chunk.output_frames
                ),
                frames_skipped=1 if chunk.pulls_successor else 0,
                bytes_out=len(payload),
                wall_seconds=self.delays.get(chunk.index, 0),
            )
        finally:
            self.concurrent -= 1


class ExecutorOrderingTest(CustomTestCase):
    def setUp(self):
        self.chunks = [
            ChunkSpec(0, "1", 0, 40, True, 2),
            ChunkSpec(1, "0", 40, 70, True, 2),
            ChunkSpec(2, "2", 70, 100, False, 2),
        ]

    def _run(self, runner, **kwargs):
        received = bytearray()

        async def sink(payload: bytes) -> None:
            received.extend(payload)

        executor = MultiCardExecutor(
            job_id="t",
            chunks=self.chunks,
            runner=runner,
            sink=sink,
            total_frames=100,
            multiplier=2,
            **kwargs,
        )
        stats = asyncio.run(executor.run())
        return bytes(received), stats

    def test_chunks_arrive_in_timeline_order_though_they_finish_reversed(self):
        """The last card finishes first; the stream must still start at chunk 0."""
        runner = FakeRunner(delays={0: 0.06, 1: 0.03, 2: 0.0})
        payload, stats = self._run(runner)
        self.assertEqual(runner.finished, [2, 1, 0])
        self.assertLess(payload.index(b"<chunk0>"), payload.index(b"<chunk1>"))
        self.assertLess(payload.index(b"<chunk1>"), payload.index(b"<chunk2>"))
        self.assertEqual(stats.state, "done")

    def test_all_cards_work_at_once(self):
        runner = FakeRunner(delays={0: 0.05, 1: 0.05, 2: 0.05})
        self._run(runner)
        self.assertEqual(runner.peak_concurrent, 3)

    def test_frames_encoded_is_the_single_card_total(self):
        runner = FakeRunner()
        _, stats = self._run(runner)
        self.assertEqual(stats.frames_encoded, expected_frame_count(100, 2))

    def test_a_chunk_that_encoded_the_wrong_count_stops_the_run(self):
        """A shard short by one frame is refused, never stitched."""
        runner = FakeRunner(frames_override={1: 59})
        with self.assertRaises(MultiCardError) as ctx:
            self._run(runner)
        self.assertIn("seam arithmetic", str(ctx.exception))

    def test_a_failing_worker_surfaces_its_stderr(self):
        class Failing(ChunkRunner):
            async def run(self, chunk, spool):
                return ChunkResult(
                    index=chunk.index,
                    card=chunk.card,
                    path=spool,
                    frames_encoded=0,
                    frames_skipped=0,
                    bytes_out=0,
                    wall_seconds=0.0,
                    error="CUDA out of memory on card 0",
                )

        with self.assertRaises(MultiCardError) as ctx:
            self._run(Failing())
        self.assertIn("out of memory", str(ctx.exception))

    def test_spool_slots_bound_how_far_ahead_the_cards_may_run(self):
        """With one slot the cards serialise; the bound is real, not advisory."""
        runner = FakeRunner(delays={0: 0.02, 1: 0.02, 2: 0.02})
        self._run(runner, spool_chunks=1)
        self.assertEqual(runner.peak_concurrent, 1)

    def test_spool_files_are_removed_after_the_run(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            self._run(runner, spool_dir=tmp)
            self.assertEqual(sorted(Path(tmp).iterdir()), [])

    def test_stats_report_per_card_time_so_balance_is_visible(self):
        runner = FakeRunner(delays={0: 0.04, 1: 0.01, 2: 0.01})
        _, stats = self._run(runner)
        snapshot = stats.snapshot()
        self.assertEqual(sorted(snapshot["cards"]), ["0", "1", "2"])
        self.assertIn("busiest_card_seconds", snapshot)
        self.assertEqual(len(snapshot["chunks"]), 3)


class WorkerEnvironmentTest(CustomTestCase):
    """A worker's card identity, which is not the same as its CUDA ordinal.

    CUDA enumerates by FASTEST_FIRST unless told otherwise; NVML enumerates by
    PCI bus. On a mixed rig those two orders disagree, and a plan built from
    NVML indices then runs its chunks on the wrong cards -- silently, because
    every card still works and the output is still correct.
    """

    def setUp(self):
        self.runner = SubprocessChunkRunner(source_url="/tmp/x.mp4", request={})
        self.chunk = ChunkSpec(0, "1", 0, 10, False, 2)

    def test_the_card_is_pinned_by_visible_devices(self):
        env = self.runner._child_env(self.chunk)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_device_order_is_forced_to_nvml_order(self):
        env = self.runner._child_env(self.chunk)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_an_inherited_device_order_does_not_win(self):
        """A stray CUDA_DEVICE_ORDER in the parent must not change the mapping."""
        runner = SubprocessChunkRunner(
            source_url="/tmp/x.mp4",
            request={},
            env={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
        )
        env = runner._child_env(self.chunk)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")


class BackPressureThroughSpoolTest(CustomTestCase):
    """A stalled consumer must stop the cards, even across the spool.

    Multi-card trades the single-card guarantee (a stall reaches the decoder
    within one ring depth) for a weaker but still bounded one: the run stops
    after at most ``spool_chunks`` completed-but-unforwarded chunks.
    """

    def test_a_stalled_sink_stops_the_workers_within_the_spool_bound(self):
        chunks = [
            ChunkSpec(i, str(i % 3), i * 10, (i + 1) * 10, i < 5, 2) for i in range(6)
        ]
        runner = FakeRunner()
        gate = asyncio.Event()

        async def scenario():
            async def sink(payload: bytes) -> None:
                await gate.wait()

            executor = MultiCardExecutor(
                job_id="stall",
                chunks=chunks,
                runner=runner,
                sink=sink,
                spool_chunks=2,
                total_frames=60,
                multiplier=2,
            )
            task = asyncio.create_task(executor.run())
            # Let everything that can run, run.
            for _ in range(50):
                await asyncio.sleep(0)
            started = len(runner.started)
            gate.set()
            await task
            return started

        started_while_stalled = asyncio.run(scenario())
        self.assertLessEqual(started_while_stalled, 2)
        self.assertEqual(len(runner.finished), 6)


if __name__ == "__main__":
    unittest.main()
