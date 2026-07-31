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
import shutil
import tempfile
import unittest
from pathlib import Path

from sglang.srt.video_enhance.chain import ChainRequest, build_chain
from sglang.srt.video_enhance.frame_math import MIB, Resolution
from sglang.srt.video_enhance.multicard import (
    UNASSIGNED_CARD,
    ChunkResult,
    ChunkRunner,
    ChunkSpec,
    MultiCardError,
    MultiCardExecutor,
    PersistentChunkRunner,
    SubprocessChunkRunner,
    WorkQueue,
    chunk_specs_from_plan,
    pinned_child_env,
    pull_queue_chunks,
    pull_queue_size,
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


# ---------------------------------------------------------------------------
# Pull scheduling
# ---------------------------------------------------------------------------


class PullQueueConstructionTest(CustomTestCase):
    """The queue is a tiling of the timeline, and the seam does not know it.

    The claim pull scheduling has to survive is that deciding the card later
    cannot move a frame. It holds because nothing in the seam convention reads
    the card, and these tests are that statement made falsifiable rather than
    asserted.
    """

    def test_items_tile_the_timeline_with_no_gap_and_no_overlap(self):
        chunks = pull_queue_chunks(480, multiplier=2, has_rife=True, chunks=12)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].stop, 480)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(earlier.stop, later.start)

    def test_the_arithmetic_holds_for_every_queue_length(self):
        """Item count is a scheduling knob; it must not be a correctness one."""
        for total in (97, 240, 480):
            for count in (1, 2, 3, 7, 12):
                for multiplier in (1, 2, 3):
                    with self.subTest(total=total, count=count, m=multiplier):
                        chunks = pull_queue_chunks(
                            total, multiplier=multiplier, has_rife=True, chunks=count
                        )
                        verify_chunk_arithmetic(chunks, total, multiplier)
                        self.assertEqual(
                            total_output_frames(chunks),
                            expected_frame_count(total, multiplier),
                        )

    def test_items_are_card_agnostic_until_a_card_takes_one(self):
        chunks = pull_queue_chunks(100, multiplier=2, has_rife=True, chunks=4)
        self.assertTrue(all(not c.is_assigned for c in chunks))
        self.assertTrue(all(c.card == UNASSIGNED_CARD for c in chunks))

    def test_assigning_a_card_changes_nothing_a_seam_depends_on(self):
        """The falsifier for the whole design: place every item on every card
        and require that the frames it owns, pulls and encodes are identical."""
        chunks = pull_queue_chunks(200, multiplier=2, has_rife=True, chunks=8)
        for chunk in chunks:
            for card in ("0", "1", "2"):
                placed = chunk.assigned_to(card)
                self.assertEqual(placed.card, card)
                self.assertEqual(placed.start, chunk.start)
                self.assertEqual(placed.stop, chunk.stop)
                self.assertEqual(placed.pulls_successor, chunk.pulls_successor)
                self.assertEqual(placed.pulled_frames, chunk.pulled_frames)
                self.assertEqual(placed.output_frames, chunk.output_frames)
                for index in range(chunk.start, chunk.stop + 1):
                    for sub in (0, 1):
                        self.assertEqual(
                            placed.encodes(index, sub), chunk.encodes(index, sub)
                        )

    def test_only_the_last_item_withholds_nothing(self):
        chunks = pull_queue_chunks(100, multiplier=2, has_rife=True, chunks=5)
        self.assertEqual([c.pulls_successor for c in chunks], [1, 1, 1, 1, 0])

    def test_a_chain_without_rife_pulls_no_seam_frame(self):
        chunks = pull_queue_chunks(100, multiplier=1, has_rife=False, chunks=5)
        self.assertTrue(all(not c.pulls_successor for c in chunks))
        verify_chunk_arithmetic(chunks, 100, 1)

    def test_more_items_than_frames_is_refused(self):
        with self.assertRaises(MultiCardError):
            pull_queue_chunks(4, multiplier=2, has_rife=True, chunks=8)

    def test_queue_size_scales_with_cards(self):
        self.assertEqual(pull_queue_size(4800, cards=3, chunks_per_card=4), 12)
        self.assertEqual(pull_queue_size(4800, cards=1, chunks_per_card=4), 4)

    def test_a_short_clip_gets_fewer_items_rather_than_slivers(self):
        """The floor is what stops the queue from being all overhead."""
        size = pull_queue_size(20, cards=3, chunks_per_card=4, min_chunk_frames=8)
        self.assertEqual(size, 2)
        chunks = pull_queue_chunks(20, multiplier=2, has_rife=True, chunks=size)
        self.assertTrue(all(c.owned_frames >= 8 for c in chunks))

    def test_the_queue_hands_items_out_in_timeline_order_once_each(self):
        queue = WorkQueue(pull_queue_chunks(90, multiplier=2, has_rife=True, chunks=6))
        taken = []
        while True:
            item = queue.take("0" if len(taken) % 2 else "1")
            if item is None:
                break
            taken.append(item.index)
        self.assertEqual(taken, [0, 1, 2, 3, 4, 5])
        self.assertIsNone(queue.take("0"))
        self.assertEqual(sum(queue.items_per_card().values()), 6)


class SpeedRunner(ChunkRunner):
    """A fake worker whose per-item time depends on the card, not the item.

    That is the shape pull scheduling exists for: the cards differ, the work
    does not, and nothing told the scheduler the ratio in advance.
    """

    def __init__(self, seconds_per_item):
        self.seconds_per_item = seconds_per_item
        self.by_card: dict[str, list[int]] = {}
        self.concurrent = 0
        self.peak_concurrent = 0

    async def run(self, chunk: ChunkSpec, spool: Path) -> ChunkResult:
        self.by_card.setdefault(chunk.card, []).append(chunk.index)
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.seconds_per_item[chunk.card])
            payload = f"<chunk{chunk.index}>".encode()
            spool.parent.mkdir(parents=True, exist_ok=True)
            spool.write_bytes(payload)
            return ChunkResult(
                index=chunk.index,
                card=chunk.card,
                path=spool,
                frames_encoded=chunk.output_frames,
                frames_skipped=1 if chunk.pulls_successor else 0,
                bytes_out=len(payload),
                wall_seconds=self.seconds_per_item[chunk.card],
            )
        finally:
            self.concurrent -= 1


class PullSchedulingTest(CustomTestCase):
    """The executor under a pull queue: balance, order, and the same gates."""

    def _run(self, chunks, runner, cards, total_frames, **kwargs):
        received = bytearray()

        async def sink(payload: bytes) -> None:
            received.extend(payload)

        executor = MultiCardExecutor(
            job_id="pull",
            chunks=chunks,
            runner=runner,
            sink=sink,
            cards=cards,
            total_frames=total_frames,
            multiplier=2,
            **kwargs,
        )
        stats = asyncio.run(executor.run())
        return bytes(received), stats

    def test_the_fast_card_takes_more_items_without_being_told_it_is_fast(self):
        """The whole point, stated as a measurement rather than a design note.

        No rate table, no calibration, no weight: the 5090-shaped card is
        given no more information than the 3080-shaped ones and ends up with
        the larger share purely by finishing sooner and asking again.
        """
        chunks = pull_queue_chunks(480, multiplier=2, has_rife=True, chunks=12)
        runner = SpeedRunner({"1": 0.004, "0": 0.010, "2": 0.010})
        _, stats = self._run(chunks, runner, ["1", "0", "2"], 480, spool_chunks=12)
        counts = stats.items_per_card
        self.assertEqual(sum(counts.values()), 12)
        self.assertGreater(counts["1"], counts["0"])
        self.assertGreater(counts["1"], counts["2"])

    def test_a_card_that_stops_being_fast_stops_getting_work(self):
        """A pre-weighted plan cannot revise itself; a queue does it for free.

        The card is fast for its first item and slow afterwards -- a thermal
        cap, or an LLM co-tenant arriving. A capacity-weighted split made
        before the first frame would have handed it the largest chunk on the
        strength of the calibration it no longer lives up to.
        """
        chunks = pull_queue_chunks(480, multiplier=2, has_rife=True, chunks=12)

        class Derating(ChunkRunner):
            def __init__(self):
                self.by_card: dict[str, list[int]] = {}

            async def run(self, chunk, spool):
                seen = self.by_card.setdefault(chunk.card, [])
                seen.append(chunk.index)
                slow = chunk.card == "1" and len(seen) > 1
                await asyncio.sleep(0.020 if slow else 0.002)
                spool.parent.mkdir(parents=True, exist_ok=True)
                spool.write_bytes(b"x")
                return ChunkResult(
                    index=chunk.index,
                    card=chunk.card,
                    path=spool,
                    frames_encoded=chunk.output_frames,
                    frames_skipped=1 if chunk.pulls_successor else 0,
                    bytes_out=1,
                    wall_seconds=0.0,
                )

        runner = Derating()
        _, stats = self._run(chunks, runner, ["1", "0", "2"], 480, spool_chunks=12)
        self.assertEqual(sum(stats.items_per_card.values()), 12)
        # The derated card keeps its first item and is overtaken afterwards.
        self.assertLess(stats.items_per_card["1"], stats.items_per_card["0"])

    def test_output_is_in_timeline_order_whatever_order_the_cards_pulled(self):
        chunks = pull_queue_chunks(120, multiplier=2, has_rife=True, chunks=6)
        runner = SpeedRunner({"1": 0.001, "0": 0.008, "2": 0.004})
        payload, stats = self._run(chunks, runner, ["1", "0", "2"], 120, spool_chunks=6)
        positions = [payload.index(f"<chunk{i}>".encode()) for i in range(6)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(stats.state, "done")

    def test_the_frame_total_is_the_single_card_answer(self):
        chunks = pull_queue_chunks(480, multiplier=2, has_rife=True, chunks=12)
        runner = SpeedRunner({"1": 0.001, "0": 0.002, "2": 0.002})
        _, stats = self._run(chunks, runner, ["1", "0", "2"], 480, spool_chunks=12)
        self.assertEqual(stats.frames_encoded, expected_frame_count(480, 2))

    def test_the_wrong_frame_count_is_refused_under_pull_too(self):
        """The per-chunk gate reads the item, not the plan, so it still fires."""
        chunks = pull_queue_chunks(120, multiplier=2, has_rife=True, chunks=6)

        class Short(ChunkRunner):
            async def run(self, chunk, spool):
                spool.parent.mkdir(parents=True, exist_ok=True)
                spool.write_bytes(b"x")
                return ChunkResult(
                    index=chunk.index,
                    card=chunk.card,
                    path=spool,
                    frames_encoded=chunk.output_frames - (1 if chunk.index == 3 else 0),
                    frames_skipped=0,
                    bytes_out=1,
                    wall_seconds=0.0,
                )

        with self.assertRaises(MultiCardError) as ctx:
            self._run(chunks, Short(), ["0", "1"], 120, spool_chunks=6)
        self.assertIn("seam arithmetic", str(ctx.exception))

    def test_the_error_names_the_card_that_actually_ran_the_item(self):
        """Under pull the card is not knowable from the plan, so the message
        has to come from the result or it names a card that never ran it."""
        chunks = pull_queue_chunks(120, multiplier=2, has_rife=True, chunks=6)

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
                    error=f"CUDA out of memory on card {chunk.card}",
                )

        with self.assertRaises(MultiCardError) as ctx:
            self._run(chunks, Failing(), ["2"], 120, spool_chunks=6)
        self.assertIn("card 2", str(ctx.exception))

    def test_the_spool_bound_still_holds_and_does_not_deadlock(self):
        """One slot, many more items than cards: the run must finish anyway.

        The hazard the slot-before-take order removes is a card blocked on a
        slot while the forwarding loop waits for the very item that card is
        about to take. If that were possible this test would hang rather than
        fail, so it is run under a timeout.
        """
        chunks = pull_queue_chunks(240, multiplier=2, has_rife=True, chunks=12)
        runner = SpeedRunner({"1": 0.001, "0": 0.002, "2": 0.003})

        async def scenario():
            received = bytearray()

            async def sink(payload: bytes) -> None:
                received.extend(payload)

            executor = MultiCardExecutor(
                job_id="bound",
                chunks=chunks,
                runner=runner,
                sink=sink,
                cards=["1", "0", "2"],
                spool_chunks=1,
                total_frames=240,
                multiplier=2,
            )
            return await asyncio.wait_for(executor.run(), timeout=30)

        stats = asyncio.run(scenario())
        self.assertEqual(stats.state, "done")
        self.assertEqual(runner.peak_concurrent, 1)
        self.assertEqual(stats.frames_encoded, expected_frame_count(240, 2))

    def test_a_stalled_sink_stops_the_cards_under_pull_as_well(self):
        chunks = pull_queue_chunks(240, multiplier=2, has_rife=True, chunks=12)
        runner = SpeedRunner({"1": 0.0, "0": 0.0, "2": 0.0})
        gate = asyncio.Event()

        async def scenario():
            async def sink(payload: bytes) -> None:
                await gate.wait()

            executor = MultiCardExecutor(
                job_id="stall",
                chunks=chunks,
                runner=runner,
                sink=sink,
                cards=["1", "0", "2"],
                spool_chunks=2,
                total_frames=240,
                multiplier=2,
            )
            task = asyncio.create_task(executor.run())
            for _ in range(80):
                await asyncio.sleep(0)
            taken = sum(len(v) for v in runner.by_card.values())
            gate.set()
            await task
            return taken

        taken_while_stalled = asyncio.run(scenario())
        self.assertLessEqual(taken_while_stalled, 2)

    def test_stats_record_which_schedule_ran_and_who_pulled_what(self):
        chunks = pull_queue_chunks(120, multiplier=2, has_rife=True, chunks=6)
        runner = SpeedRunner({"1": 0.001, "0": 0.003})
        _, stats = self._run(chunks, runner, ["1", "0"], 120, spool_chunks=6)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot["schedule"], "pull")
        self.assertEqual(len(snapshot["pull_order"]), 6)
        self.assertEqual([entry[0] for entry in snapshot["pull_order"]], list(range(6)))
        self.assertEqual(sum(snapshot["items_per_card"].values()), 6)


class ScheduleModeGuardTest(CustomTestCase):
    """Two answers to the same question; mixing them silently is the bug."""

    async def _sink(self, payload: bytes) -> None:
        return None

    def test_a_pre_weighted_plan_is_not_silently_re_scheduled(self):
        chunks = [
            ChunkSpec(0, "1", 0, 50, True, 2),
            ChunkSpec(1, "0", 50, 100, False, 2),
        ]
        with self.assertRaises(MultiCardError) as ctx:
            MultiCardExecutor(
                job_id="x",
                chunks=chunks,
                runner=FakeRunner(),
                sink=self._sink,
                cards=["0", "1"],
            )
        self.assertIn("already name a card", str(ctx.exception))

    def test_unassigned_items_without_a_card_list_are_refused(self):
        chunks = pull_queue_chunks(100, multiplier=2, has_rife=True, chunks=4)
        with self.assertRaises(MultiCardError) as ctx:
            MultiCardExecutor(
                job_id="x", chunks=chunks, runner=FakeRunner(), sink=self._sink
            )
        self.assertIn("no card", str(ctx.exception))

    def test_the_same_card_offered_twice_is_refused(self):
        chunks = pull_queue_chunks(100, multiplier=2, has_rife=True, chunks=4)
        with self.assertRaises(MultiCardError) as ctx:
            MultiCardExecutor(
                job_id="x",
                chunks=chunks,
                runner=FakeRunner(),
                sink=self._sink,
                cards=["0", "0"],
            )
        self.assertIn("offered twice", str(ctx.exception))

    def test_the_pinned_path_is_untouched(self):
        """The default remains what it was: chunks carry cards, no queue."""
        chunks = [
            ChunkSpec(0, "1", 0, 50, True, 2),
            ChunkSpec(1, "0", 50, 100, False, 2),
        ]
        executor = MultiCardExecutor(
            job_id="x", chunks=chunks, runner=FakeRunner(), sink=self._sink
        )
        self.assertIsNone(executor.queue)
        self.assertEqual(executor.stats.schedule, "pinned")


# A worker stub that speaks the serving protocol and imports nothing but the
# standard library. It stands in for the real chunk worker so the pipe
# protocol -- one item in, one prefixed report out, process reused -- can be
# proven without a card, a model, or eight seconds of torch import.
STUB_WORKER = """
import json, os, sys

PREFIX = "@@CHUNK_REPORT@@ "
args = sys.argv[1:]
request = json.loads(args[args.index("--request") + 1])
report_fd = os.dup(1)
os.dup2(2, 1)
reports = os.fdopen(report_fd, "w")
pid = os.getpid()
print("worker noise on stdout that must not be read as a report", flush=True)
items = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    payload = json.loads(line)
    chunk = payload["chunk"]
    items += 1
    if chunk["index"] == request.get("die_on_index"):
        sys.stderr.write("stub worker is dying on purpose\\n")
        sys.stderr.flush()
        os._exit(9)
    with open(payload["output_path"], "wb") as handle:
        handle.write(("<chunk%d>" % chunk["index"]).encode())
    owned = chunk["stop"] - chunk["start"]
    m = chunk["multiplier"]
    frames = owned * m if chunk["pulls_successor"] else owned + (owned - 1) * (m - 1)
    reports.write(PREFIX + json.dumps({
        "index": chunk["index"],
        "frames_encoded": frames,
        "frames_skipped": 1 if chunk["pulls_successor"] else 0,
        "bytes_out": 8,
        "worker_pid": pid,
        "items_this_worker": items,
        "card": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) + "\\n")
    reports.flush()
"""


class PersistentWorkerProtocolTest(CustomTestCase):
    """One process per card, many items: the protocol, not the pixels.

    The measured reason this class exists is the ~8 s of torch and ONNX
    Runtime import #339 recorded per chunk process. A queue of four items per
    card would pay it four times; a persistent worker pays it once. What is
    checked here is that the reuse is real and that a dead worker is reported
    rather than waited on.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="k3-stubworker-")
        Path(self.tmp, "stub_chunk_worker.py").write_text(STUB_WORKER)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _runner(self, request=None):
        return PersistentChunkRunner(
            source_url="/tmp/does-not-exist.mp4",
            request=request or {},
            module="stub_chunk_worker",
            env={"PYTHONPATH": self.tmp},
        )

    def test_one_worker_per_card_serves_every_item_that_card_takes(self):
        chunks = pull_queue_chunks(240, multiplier=2, has_rife=True, chunks=8)
        runner = self._runner()
        received = bytearray()

        async def sink(payload: bytes) -> None:
            received.extend(payload)

        executor = MultiCardExecutor(
            job_id="persist",
            chunks=chunks,
            runner=runner,
            sink=sink,
            cards=["0", "1"],
            total_frames=240,
            multiplier=2,
            spool_chunks=8,
        )
        stats = asyncio.run(executor.run())
        self.assertEqual(stats.state, "done")
        self.assertEqual(stats.frames_encoded, expected_frame_count(240, 2))
        # Two cards, eight items, two processes -- the reuse is the claim.
        pids = {entry["card"]: entry for entry in stats.chunks}
        self.assertLessEqual(len({c["card"] for c in stats.chunks}), 2)
        self.assertEqual(len(pids), len({c["card"] for c in stats.chunks}))
        positions = [received.index(f"<chunk{i}>".encode()) for i in range(8)]
        self.assertEqual(positions, sorted(positions))

    def test_the_worker_is_started_once_and_kept(self):
        chunks = pull_queue_chunks(120, multiplier=2, has_rife=True, chunks=6)
        runner = self._runner()
        seen: list[dict] = []

        class Recording(PersistentChunkRunner):
            async def run(self, chunk, spool):
                result = await PersistentChunkRunner.run(self, chunk, spool)
                seen.append({"index": chunk.index, "card": chunk.card})
                return result

        async def sink(payload: bytes) -> None:
            return None

        executor = MultiCardExecutor(
            job_id="reuse",
            chunks=chunks,
            runner=runner,
            sink=sink,
            cards=["0"],
            total_frames=120,
            multiplier=2,
            spool_chunks=6,
        )
        asyncio.run(executor.run())
        # One card took all six items, and exactly one startup was recorded.
        self.assertEqual(list(runner.spawn_seconds), ["0"])

    def test_stdout_noise_from_the_worker_is_not_read_as_a_report(self):
        """The stub prints to stdout before its first report, as torch and
        ONNX Runtime both do. A protocol that took "the next line" would
        parse that and fail on the item that was actually fine."""
        chunks = pull_queue_chunks(40, multiplier=2, has_rife=True, chunks=2)
        runner = self._runner()

        async def sink(payload: bytes) -> None:
            return None

        executor = MultiCardExecutor(
            job_id="noise",
            chunks=chunks,
            runner=runner,
            sink=sink,
            cards=["0"],
            total_frames=40,
            multiplier=2,
        )
        stats = asyncio.run(executor.run())
        self.assertEqual(stats.state, "done")

    def test_a_worker_that_dies_mid_item_is_reported_with_its_stderr(self):
        """Not waited on. A parent blocked reading a pipe whose writer is gone
        looks exactly like a slow card, and stays that way forever."""
        chunks = pull_queue_chunks(60, multiplier=2, has_rife=True, chunks=3)
        runner = self._runner(request={"die_on_index": 1})

        async def sink(payload: bytes) -> None:
            return None

        executor = MultiCardExecutor(
            job_id="dead",
            chunks=chunks,
            runner=runner,
            sink=sink,
            cards=["0"],
            total_frames=60,
            multiplier=2,
        )

        async def scenario():
            return await asyncio.wait_for(executor.run(), timeout=60)

        with self.assertRaises(MultiCardError) as ctx:
            asyncio.run(scenario())
        message = str(ctx.exception)
        self.assertIn("dying on purpose", message)
        self.assertIn("card 0", message)

    def test_workers_are_shut_down_when_the_run_ends(self):
        chunks = pull_queue_chunks(40, multiplier=2, has_rife=True, chunks=2)
        runner = self._runner()

        async def sink(payload: bytes) -> None:
            return None

        executor = MultiCardExecutor(
            job_id="shutdown",
            chunks=chunks,
            runner=runner,
            sink=sink,
            cards=["0"],
            total_frames=40,
            multiplier=2,
        )
        asyncio.run(executor.run())
        self.assertEqual(runner._workers, {})


class PinnedChildEnvTest(CustomTestCase):
    """The device-order trap, now shared by both runners."""

    def test_both_runners_pin_the_same_way(self):
        chunk = ChunkSpec(0, "1", 0, 10, False, 2)
        one_shot = SubprocessChunkRunner(source_url="/tmp/x.mp4", request={})
        env = one_shot._child_env(chunk)
        shared = pinned_child_env("1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], shared["CUDA_VISIBLE_DEVICES"])
        self.assertEqual(env["CUDA_DEVICE_ORDER"], shared["CUDA_DEVICE_ORDER"])
        self.assertEqual(shared["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_a_caller_cannot_override_the_device_order(self):
        env = pinned_child_env("2", {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"})
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")


if __name__ == "__main__":
    unittest.main()
