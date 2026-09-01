"""#809/W28 slice 2: the DEVICE SIDE of chunk-rotation residency.

Slice 1 (`rotation_plan.py`) fixed the arithmetic: how big the overshoot is and
in what order the chunks go. This is the executor that RUNS that plan, and it
exists because the plan alone is not sufficient on this arena.

THE HAZARD THE PLAN DOES NOT ENCODE, and it is why the ring is load-bearing
rather than an optimisation. The arena is ONE contiguous device tensor sized
`max(pp, tp)` (`weights_arena.allocate_arena`), and the refill overwrites
`arena[: layout.total_bytes]` IN PLACE (`weights_arena.py:1184,1192`). The host
image is likewise one buffer reused for whichever layout is resting. So at any
chunk offset k the two directions are CIRCULARLY dependent:

    the H2D wants to write arena[k]   -- which the D2H still has to read
    the D2H wants to write image[k]   -- which the H2D still has to read

Serialising them destroys the duplex the whole scheme is for; running them
concurrently corrupts the image. The ring breaks the cycle by holding the
incoming chunk while the outgoing one is placed:

    save  image[k] -> ring slot          (host)
    D2H   arena[k] -> image[k]           (lane D)
    H2D   ring slot -> arena[k]          (lane H, gated on that D2H)

Chunk k+1's D2H then overlaps chunk k's H2D on the other lane, which is a real
full-duplex pipeline with no aliasing anywhere in it.

AND THE RAM ARITHMETIC THEN COMES OUT EXACTLY AS SLICE 1 PRICED IT: one
max-sized host buffer (= one image PLUS the size asymmetry) plus `depth *
chunk` of ring = `rotation_overshoot_bytes`. The two slices agree by
construction rather than by coincidence.

Every test here runs the REAL executor over REAL byte patterns on CPU tensors,
so byte-exactness, the checksum and the absence of a leak are executed rather
than modelled. Only the CUDA lane mapping needs metal.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool
from sglang.srt.model_executor.rotation_executor import (
    RotationHazard,
    RotationStats,
    TorchRotationOps,
    rotate_arena,
)
from sglang.srt.model_executor.rotation_plan import rotation_overshoot_bytes
from sglang.srt.model_executor.weights_arena import (
    _CHECKSUM_BYTES,
    RefillLegTiming,
    refill_bound_phrase,
    uint8_checksum,
)
from sglang.test.test_utils import CustomTestCase

CHUNK = 4096
DEPTH = 4

#: Deliberately NOT round multiples of the chunk: the tails are where an
#: off-by-one in the plan/executor handshake would hide.
INCOMING = 9 * CHUNK + 137
OUTGOING = 11 * CHUNK + 41


def _ring(chunk_bytes=CHUNK, depth=DEPTH):
    # register=False: these are test buffers and must not be charged to the
    # process-wide pinned-host ledger, which is a real registry.
    return ReadBufferPool(
        name="test_rotation_ring",
        flag="--test",
        capacity=depth,
        page_bytes=chunk_bytes,
        factory=lambda: torch.empty(chunk_bytes, dtype=torch.uint8),
        register=False,
    )


def _pattern(nbytes: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (nbytes,), generator=g, dtype=torch.uint8)


def _fresh(incoming=INCOMING, outgoing=OUTGOING):
    """An arena holding the OUTGOING layout and a host image holding the
    INCOMING one, both max-sized, exactly as boot leaves them."""
    span = max(incoming, outgoing)
    arena = torch.zeros(span, dtype=torch.uint8)
    image = torch.zeros(span + _CHECKSUM_BYTES, dtype=torch.uint8)
    out_bytes = _pattern(outgoing, seed=11)
    in_bytes = _pattern(incoming, seed=22)
    arena[:outgoing] = out_bytes
    image[:incoming] = in_bytes
    # The image's trailer describes the INCOMING image it currently holds.
    in_sum = torch.tensor([uint8_checksum(image[:incoming])], dtype=torch.int64)
    image[incoming : incoming + _CHECKSUM_BYTES] = in_sum.view(torch.uint8)
    return arena, image, in_bytes, out_bytes


class TestTheRotationIsByteExact(CustomTestCase):
    """The first thing it must do is be correct, in both directions."""

    def test_the_arena_ends_with_the_incoming_image_and_ram_with_the_outgoing(self):
        for label, (inc, out) in (
            ("outgoing_larger", (INCOMING, OUTGOING)),
            ("incoming_larger", (OUTGOING, INCOMING)),
        ):
            with self.subTest(direction=label):
                arena, image, in_bytes, out_bytes = _fresh(inc, out)
                stats = rotate_arena(
                    arena=arena,
                    host_image=image,
                    incoming_bytes=inc,
                    outgoing_bytes=out,
                    chunk_bytes=CHUNK,
                    depth=DEPTH,
                    ring=_ring(),
                    ops=TorchRotationOps(),
                )
                self.assertTrue(torch.equal(arena[:inc], in_bytes))
                self.assertTrue(torch.equal(image[:out], out_bytes))
                self.assertEqual(stats.h2d_bytes, inc)
                self.assertEqual(stats.d2h_bytes, out)

    def test_a_returned_image_verifies_by_the_arena_checksum(self):
        # Falsifier 3. The image is raw arena bytes plus an int64 trailer, so
        # bytes that came back from the device must verify exactly the way
        # bytes that came off disk do.
        arena, image, _in_bytes, out_bytes = _fresh()
        want = uint8_checksum(arena[:OUTGOING])
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
        )
        got = int(
            image[OUTGOING : OUTGOING + _CHECKSUM_BYTES]
            .clone()
            .view(torch.int64)
            .item()
        )
        self.assertEqual(got, want)
        self.assertEqual(uint8_checksum(image[:OUTGOING]), want)
        self.assertTrue(torch.equal(image[:OUTGOING], out_bytes))

    def test_a_corrupted_return_is_CAUGHT_by_that_checksum(self):
        # The can-fail partner: the check above is only evidence if it can
        # fail. Flip one byte and the trailer must no longer describe it.
        arena, image, _i, _o = _fresh()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
        )
        stored = int(
            image[OUTGOING : OUTGOING + _CHECKSUM_BYTES]
            .clone()
            .view(torch.int64)
            .item()
        )
        image[7] = (int(image[7]) + 1) % 256
        self.assertNotEqual(uint8_checksum(image[:OUTGOING]), stored)


class TestTheDuplexActuallyOverlaps(CustomTestCase):
    """WHAT THIS CLASS ACTUALLY PINS -- retracted claim kept in place (#1082).

    It used to read: "Falsifier 1, measured on the EXECUTOR rather than on the
    plan ... at the instant a chunk's D2H is enqueued, an earlier chunk's H2D
    had not yet been waited on." The mechanism description is right; the word
    FALSIFIER is not, and the class name is now wrong on purpose rather than by
    accident -- it is left alone so log and test greps still find it.

    ``overlapped_steps`` is reproducible from (h2d chunks, d2h chunks, depth)
    by ``plan_determined_overlap``: exact on 12 of 12 production legs, and the
    same value on a 4 s leg and a 41 s leg of identical shape. So these two
    tests pin a PIPELINING DECISION of the executor -- real, and worth pinning
    -- and NOT that the two directions overlapped on the device. The
    ``depth=1`` case below reads as a can-fail partner and is not one: it varies
    with a CONFIG input, which every plan-shaped counter also does.

    The duplex premise is answered by ``RotationPhases.gpu_d2h_s`` /
    ``gpu_h2d_s``, which had no writer at all until #1082.
    """

    def test_a_real_rotation_overlaps_on_almost_every_step(self):
        arena, image, _i, _o = _fresh()
        stats = rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
        )
        self.assertGreater(stats.overlapped_steps, 0)
        # DOMINANT, not incidental: every step but the first has an earlier
        # H2D still outstanding when its copy-back is enqueued. Asserted as
        # `steps - 1` rather than a percentage because that is the exact
        # shape, and a percentage would drift with the chunk count.
        self.assertEqual(stats.overlapped_steps, stats.steps - 1)
        self.assertGreater(stats.overlap_share, 0.9)

    def test_depth_one_CANNOT_overlap_and_says_so(self):
        # THE CAN-FAIL PARTNER. A ring of one slot must be drained before the
        # next chunk can be saved, so nothing is ever in flight across steps.
        # A counter that still reported overlap here would be measuring the
        # plan's shape instead of the executor's behaviour.
        arena, image, in_bytes, out_bytes = _fresh()
        stats = rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=1,
            ring=_ring(depth=1),
            ops=TorchRotationOps(),
        )
        self.assertEqual(stats.overlapped_steps, 0)
        # ...and it is still CORRECT, only slow. A serialised rotation must
        # not be a broken one.
        self.assertTrue(torch.equal(arena[:INCOMING], in_bytes))
        self.assertTrue(torch.equal(image[:OUTGOING], out_bytes))


class TestTheAliasingHazardIsRefusedNotRisked(CustomTestCase):
    """The defect Slice 1's arithmetic cannot see.

    Both directions touch the same offsets of the same two buffers. Without a
    ring to hold the incoming chunk, the H2D would read host bytes the D2H had
    already overwritten -- silent corruption of the NEXT flip's image, which is
    exactly the failure mode a checksum on THIS flip would not catch.
    """

    def test_an_overlapping_rotation_without_a_ring_is_REFUSED(self):
        arena, image, _i, _o = _fresh()
        with self.assertRaises(RotationHazard) as caught:
            rotate_arena(
                arena=arena,
                host_image=image,
                incoming_bytes=INCOMING,
                outgoing_bytes=OUTGOING,
                chunk_bytes=CHUNK,
                depth=DEPTH,
                ring=None,
                ops=TorchRotationOps(),
            )
        self.assertIn("in place", str(caught.exception).lower())

    def test_a_one_sided_rotation_needs_no_ring_and_is_ALLOWED(self):
        # THE CAN-FAIL PARTNER, and it is the priming flip's shape: nothing to
        # copy back, so no offset is aliased and the refusal must not fire.
        # A guard that refused here would block the very first flip.
        arena, image, in_bytes, _o = _fresh()
        stats = rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=0,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=None,
            ops=TorchRotationOps(),
        )
        self.assertEqual(stats.d2h_bytes, 0)
        self.assertEqual(stats.overlapped_steps, 0)
        self.assertTrue(torch.equal(arena[:INCOMING], in_bytes))


class TestNoLeakAndNoDriftAcrossCycles(CustomTestCase):
    """Falsifier 2, over THREE full cycles because W27-retry's leak fired on
    the third rather than the first -- and here the bytes are checked too, not
    only the occupancy."""

    def test_three_cycles_return_both_buffers_to_their_starting_content(self):
        arena, image, in_bytes, out_bytes = _fresh()
        ring = _ring()
        ops = TorchRotationOps()
        start_ram = image.numel()
        inc, out = INCOMING, OUTGOING
        for _cycle in range(3):
            for a, b in ((inc, out), (out, inc)):
                rotate_arena(
                    arena=arena,
                    host_image=image,
                    incoming_bytes=a,
                    outgoing_bytes=b,
                    chunk_bytes=CHUNK,
                    depth=DEPTH,
                    ring=ring,
                    ops=ops,
                )
            # One A->B->A cycle must restore BOTH sides bit for bit.
            self.assertTrue(torch.equal(arena[:out], out_bytes))
            self.assertTrue(torch.equal(image[:inc], in_bytes))
            # And the host buffer must not have grown to do it.
            self.assertEqual(image.numel(), start_ram)
        self.assertEqual(ring.overflow_allocations, 0, "the ring spilled its bound")

    def test_the_ring_is_returned_empty_so_a_later_flip_finds_its_slots(self):
        # A slot leaked per flip is a ring that silently degrades into
        # unbounded allocation by the third rotation -- the shape of the leak
        # this class exists to catch, one level down.
        ring = _ring()
        arena, image, _i, _o = _fresh()
        for a, b in ((INCOMING, OUTGOING), (OUTGOING, INCOMING)) * 3:
            rotate_arena(
                arena=arena,
                host_image=image,
                incoming_bytes=a,
                outgoing_bytes=b,
                chunk_bytes=CHUNK,
                depth=DEPTH,
                ring=ring,
                ops=TorchRotationOps(),
            )
            self.assertEqual(ring.available, DEPTH)
            self.assertEqual(ring.overflow_allocations, 0)


class TestTheIncomingImageIsVerifiedBeforeItIsServed(CustomTestCase):
    """The arena-side half of falsifier 3, and it must be able to fire.

    ``arena_refill`` already verifies a streamed-in image against its trailer;
    the rotation must not lose that check just because the bytes came from RAM
    rather than from disk.
    """

    def test_an_image_whose_trailer_does_not_describe_it_is_REFUSED(self):
        arena, image, _i, _o = _fresh()
        image[3] = (int(image[3]) + 1) % 256  # payload no longer sums to the trailer
        with self.assertRaises(RotationHazard) as caught:
            rotate_arena(
                arena=arena,
                host_image=image,
                incoming_bytes=INCOMING,
                outgoing_bytes=OUTGOING,
                chunk_bytes=CHUNK,
                depth=DEPTH,
                ring=_ring(),
                ops=TorchRotationOps(),
            )
        self.assertIn("checksum mismatch", str(caught.exception))

    def test_an_intact_image_passes_that_same_check(self):
        # THE CAN-FAIL PARTNER: a guard that refused every image would be
        # indistinguishable from one that works, and would block every flip.
        arena, image, in_bytes, _o = _fresh()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
        )
        self.assertTrue(torch.equal(arena[:INCOMING], in_bytes))


class TestThePrimingFlipIsInstrumentedApart(CustomTestCase):
    """P4: the first flip after boot still primes from disk. A steady-state
    claim averaged over a mean that includes it is the measurement error this
    ticket is most likely to make, so the two must not share a record."""

    def test_a_priming_rotation_is_marked_and_a_warm_one_is_not(self):
        arena, image, _i, _o = _fresh()
        primed = rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=0,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=None,
            ops=TorchRotationOps(),
            priming=True,
        )
        self.assertTrue(primed.priming)
        arena, image, _i, _o = _fresh()
        warm = rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
        )
        self.assertFalse(warm.priming)

    def test_a_warm_rotation_never_folds_priming_time_into_its_timing(self):
        # Two records, never one. Handing the same RefillLegTiming to both
        # would produce exactly the polluted mean P4 forbids.
        prime_t, warm_t = RefillLegTiming(), RefillLegTiming()
        arena, image, _i, _o = _fresh()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=0,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=None,
            ops=TorchRotationOps(),
            timing=prime_t,
            priming=True,
        )
        arena, image, _i, _o = _fresh()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
            timing=warm_t,
        )
        self.assertGreater(prime_t.chunks, 0)
        self.assertGreater(warm_t.chunks, 0)
        self.assertNotEqual(prime_t.chunks, warm_t.chunks)


class TestTheLegNamesItsBound(CustomTestCase):
    """Reuse the #856(a) instrument rather than build new telemetry: the
    acceptance readout for W28 is `refill_bound_phrase` over this leg.

    #1082 RETRACTION, in the class that carried the claim. Reusing that
    instrument here was wrong in a way this file could not see: the phrase
    weighs read against h2d-wait, and on THIS path ``read_s`` has no writer at
    all (its only one lives in ``_staged_file_refill``, which the rotation does
    not call). So the phrase was structurally incapable of returning anything
    but LINK-BOUND, and the test below asserted exactly that outcome as the
    ticket's acceptance -- a green that could not have gone red. On metal it
    printed `LINK-BOUND (read 0.000s / h2d-wait 0.016s)` on a 63.931 s leg.
    The phrase now refuses below ``BOUND_MIN_COVERAGE``, and these tests assert
    the refusal instead of the verdict.
    """

    def test_an_instrumented_rotation_is_not_unattributed(self):
        timing = RefillLegTiming()
        arena, image, _i, _o = _fresh()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
            timing=timing,
        )
        self.assertGreater(timing.chunks, 0)
        self.assertNotIn("unattributed", refill_bound_phrase(timing))

    def test_an_uninstrumented_rotation_still_says_unattributed(self):
        # THE CAN-FAIL PARTNER: the phrase must keep its honest answer for a
        # leg nobody measured, or "LINK-BOUND" becomes unfalsifiable.
        self.assertIn("unattributed", refill_bound_phrase(RefillLegTiming()))

    def test_a_rotation_reads_no_storage_which_is_the_whole_point(self):
        # W28's acceptance criterion, CORRECTED (#1082). The old comment read:
        # "a WARM rotation touches no disk, so its read_s is zero and the bound
        # must land on the link." The first half is the real acceptance and is
        # still asserted below. The second half was a non-sequitur: read_s is
        # zero on this path whether the rotation touched disk or not, because
        # nothing on this path ever writes it. "The bound lands on the link"
        # therefore followed from the instrument's wiring, not from the leg.
        # What the leg is actually bound by is now an open question with an
        # instrument attached -- see RotationPhases.gpu_d2h_s / gpu_h2d_s.
        timing = RefillLegTiming()
        arena, image, _i, _o = _fresh()
        rotate_arena(
            arena=arena,
            host_image=image,
            incoming_bytes=INCOMING,
            outgoing_bytes=OUTGOING,
            chunk_bytes=CHUNK,
            depth=DEPTH,
            ring=_ring(),
            ops=TorchRotationOps(),
            timing=timing,
        )
        self.assertEqual(timing.read_s, 0.0)


class TestTheBudgetLawSurvivesTheExecutor(CustomTestCase):
    """R1's directional finding is law and must not regress through the
    executor: pressure exists ONLY when the outgoing layout is the larger."""

    def test_the_executor_asks_for_exactly_the_slice_one_overshoot(self):
        over = rotation_overshoot_bytes(INCOMING, OUTGOING, CHUNK, DEPTH)
        self.assertEqual(over, abs(INCOMING - OUTGOING) + DEPTH * CHUNK)
        # The host buffer the executor requires is one max-sized image plus the
        # ring, and that is the same number -- the two slices agree by
        # construction.
        span = max(INCOMING, OUTGOING) + DEPTH * CHUNK
        self.assertEqual(span, min(INCOMING, OUTGOING) + over)

    def test_a_host_buffer_too_small_for_the_larger_layout_is_REFUSED(self):
        arena, image, _i, _o = _fresh()
        short = image[: min(INCOMING, OUTGOING)].clone()
        with self.assertRaises(RotationHazard):
            rotate_arena(
                arena=arena,
                host_image=short,
                incoming_bytes=INCOMING,
                outgoing_bytes=OUTGOING,
                chunk_bytes=CHUNK,
                depth=DEPTH,
                ring=_ring(),
                ops=TorchRotationOps(),
            )


class TestTheHostPostIsPricedNotHandPinned(CustomTestCase):
    """P3: the ring is a HOST POST and must reach the ledger, not be pinned by
    hand beside it. Priced at the launcher (a pure function), registered in the
    worker that actually allocates it (#720 ring over #729's
    register-then-allocate) -- never registered at the launcher, which is the
    helper commit 272d0d9d8c deleted."""

    def test_the_host_cost_is_ONE_image_plus_the_ring(self):
        from sglang.srt.model_executor.rotation_executor import rotation_host_bytes

        image, ring = rotation_host_bytes(INCOMING, OUTGOING, CHUNK, DEPTH)
        self.assertEqual(image, max(INCOMING, OUTGOING) + _CHECKSUM_BYTES)
        self.assertEqual(ring, CHUNK * DEPTH)
        # The point of the whole scheme: NOT the sum of both layouts, which is
        # the dual pin W26 proved impossible (both arms OOM-killed at launch).
        self.assertLess(image + ring, INCOMING + OUTGOING)

    def test_the_ring_is_registered_ONCE_and_reused(self):
        from sglang.srt.model_executor import rotation_executor as rx

        saved = rx._rotation_ring
        try:
            rx._rotation_ring = None
            with unittest.mock.patch.object(
                rx, "_ring_lock", __import__("threading").Lock()
            ):
                made = []
                real = ReadBufferPool

                def _spy(**kw):
                    kw["register"] = False
                    kw["factory"] = lambda: torch.empty(
                        kw["page_bytes"], dtype=torch.uint8
                    )
                    made.append(kw["name"])
                    return real(**kw)

                with unittest.mock.patch(
                    "sglang.srt.mem_cache.read_buffer_pool.ReadBufferPool", _spy
                ):
                    a = rx.rotation_ring(CHUNK, DEPTH)
                    b = rx.rotation_ring(CHUNK, DEPTH)
                self.assertIs(a, b, "a second flip rebuilt the ring")
                self.assertEqual(len(made), 1)
        finally:
            rx._rotation_ring = saved

    def test_the_launcher_PRICES_the_ring_and_refuses_when_it_cannot_fit(self):
        from sglang.srt import server_args as sa

        args = sa.ServerArgs.__new__(sa.ServerArgs)
        args.tp_size, args.pp_size = 3, 1
        with unittest.mock.patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            return_value=(1024, 1024),  # a machine with 1 KiB to pin
        ):
            with self.assertRaises(ValueError) as caught:
                args._post_phase_flip_rotation_host_ledger()
        self.assertIn("rotation staging ring", str(caught.exception))

    def test_it_does_NOT_refuse_on_a_machine_that_has_the_room(self):
        # THE CAN-FAIL PARTNER: a check that refused every boot would be
        # indistinguishable from one that works.
        from sglang.srt import server_args as sa

        args = sa.ServerArgs.__new__(sa.ServerArgs)
        args.tp_size, args.pp_size = 3, 1
        with unittest.mock.patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            return_value=(1 << 40, 1 << 40),
        ):
            args._post_phase_flip_rotation_host_ledger()


class TestStatsAreARecordNotAJudgement(CustomTestCase):
    def test_the_stats_object_is_a_plain_record(self):
        s = RotationStats()
        self.assertEqual(s.steps, 0)
        self.assertEqual(s.overlapped_steps, 0)
        self.assertFalse(s.priming)


if __name__ == "__main__":
    unittest.main()
