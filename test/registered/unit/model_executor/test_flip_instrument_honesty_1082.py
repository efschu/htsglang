"""#1082: three instruments on the flip's weights leg reported things that
were not measured, and two code comments cited one of them as evidence.

THE THREE, each found by reading the code against the boot it produced
(boot_855_1078spec_1677a9d463_0901_155207, six legs, whole file):

  1. ``RotationPhases.gpu_d2h_s`` / ``gpu_h2d_s`` had exactly ONE writer in the
     entire tree -- their own dataclass default -- and one reader, the phase
     renderer. The field docstring promised "read ONCE after the drain"; no
     line ever read them. Every phase line in every boot printed
     `gpu-span d2h 0.000s / h2d 0.000s`, a hard-coded zero, and
     phase_flip_boot.py cited that zero twice as proof that no device transfer
     took part in the leg.

  2. ``d2h_issue_s`` measured a completed transfer, not an issue. The D2H
     destination is the host image, which under the file-backed arm is a
     pageable shared mapping, so ``copy_(non_blocking=True)`` is synchronous
     with respect to the host. The term was 94.7-96.1 % of the leg on all six
     legs while being named after an enqueue.

  3. ``refill_bound_phrase`` returned LINK-BOUND on this path and could return
     nothing else: it weighs ``read_s`` against ``h2d_wait_s``, and ``read_s``
     has no writer on this path at all. Covered in test_refill_bound_856.py.

AND A FOURTH, from R-TIME: ``overlapped_steps`` is reproducible in closed form
from (h2d chunks, d2h chunks, depth) -- exact on 12 of 12 production legs, and
identical on a 4 s and a 41 s leg of the same shape. It is a pipelining
decision, not evidence that the two directions overlapped on the device. Its
docstring claimed the opposite in as many words.

THE CAN-FAIL SHAPE THIS FILE IS BUILT AROUND: every assertion here must go RED
under the mutant that RE-BLINDS the instrument, because "blind" is precisely
the state all four were shipped in. For the spans that means a planted zero has
to be distinguishable from a real reading -- which is why they are Optional and
render as `not-measured`, and why `0.000s` in a phase line is now a claim.

Runs on CPU: the wiring, the Optional contract and the closed form are all
device-free. Only the span VALUES need metal, and the span-carrying test
injects them so the wiring is provable without one.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool
from sglang.srt.model_executor.rotation_executor import (
    RotationPhases,
    RotationStats,
    TorchRotationOps,
    plan_determined_overlap,
    rotate_arena,
    rotation_phase_report,
    rotation_report,
)
from sglang.srt.model_executor.weights_arena import _CHECKSUM_BYTES, uint8_checksum
from sglang.test.test_utils import CustomTestCase

CHUNK = 4096
DEPTH = 4
INCOMING = 9 * CHUNK + 137
OUTGOING = 11 * CHUNK + 41


def _ring(chunk_bytes=CHUNK, depth=DEPTH):
    return ReadBufferPool(
        name="test_1082_ring",
        flag="--test",
        capacity=depth,
        page_bytes=chunk_bytes,
        factory=lambda: torch.empty(chunk_bytes, dtype=torch.uint8),
        register=False,
    )


def _fresh(incoming=INCOMING, outgoing=OUTGOING):
    span = max(incoming, outgoing)
    arena = torch.zeros(span, dtype=torch.uint8)
    image = torch.zeros(span + _CHECKSUM_BYTES, dtype=torch.uint8)
    g = torch.Generator().manual_seed(11)
    arena[:outgoing] = torch.randint(
        0, 256, (outgoing,), generator=g, dtype=torch.uint8
    )
    g2 = torch.Generator().manual_seed(22)
    image[:incoming] = torch.randint(
        0, 256, (incoming,), generator=g2, dtype=torch.uint8
    )
    image[incoming : incoming + _CHECKSUM_BYTES] = torch.tensor(
        [uint8_checksum(image[:incoming])], dtype=torch.int64
    ).view(torch.uint8)
    return arena, image


def _rotate(phases=None, ops=None, depth=DEPTH, incoming=INCOMING, outgoing=OUTGOING):
    arena, image = _fresh(incoming, outgoing)
    return rotate_arena(
        arena=arena,
        host_image=image,
        incoming_bytes=incoming,
        outgoing_bytes=outgoing,
        chunk_bytes=CHUNK,
        depth=depth,
        ring=_ring(depth=depth),
        ops=ops if ops is not None else TorchRotationOps(),
        phases=phases,
    )


class _SpanInjectingOps(TorchRotationOps):
    """The REAL executor with only ``device_spans`` replaced.

    Subclassed rather than stubbed on purpose: the byte movement, the ring and
    the aliasing logic stay real, so this test can only fail on the one thing it
    is about -- whether ``rotate_arena`` actually READS the spans and stores
    them on the phases object.
    """

    INJECTED = (1.25, 0.75)

    def __init__(self):
        super().__init__()
        self.span_reads = 0

    def device_spans(self):
        self.span_reads += 1
        return self.INJECTED


class TestTheDeviceSpansAreActuallyRead(CustomTestCase):
    """Posten 1: the fields had no writer. Now they have exactly one."""

    def test_the_rotation_stores_the_spans_it_was_given(self):
        # THE MUTANT THIS KILLS: delete the `phases.gpu_d2h_s, phases.gpu_h2d_s
        # = ops.device_spans()` line in rotate_arena -- i.e. restore the state
        # the tree shipped in -- and both fields stay None here.
        ops = _SpanInjectingOps()
        phases = RotationPhases()
        _rotate(phases=phases, ops=ops)
        self.assertEqual(ops.span_reads, 1, "spans must be read exactly ONCE")
        self.assertEqual(phases.gpu_d2h_s, _SpanInjectingOps.INJECTED[0])
        self.assertEqual(phases.gpu_h2d_s, _SpanInjectingOps.INJECTED[1])

    def test_the_spans_are_read_once_not_per_chunk(self):
        # The docstring's promise is "read ONCE after the drain, never
        # synchronised inside the loop" (the ms-per-round canon). A per-chunk
        # read would sync the lanes and destroy the very overlap it measures.
        ops = _SpanInjectingOps()
        stats = _rotate(phases=RotationPhases(), ops=ops)
        self.assertGreater(stats.steps, 10)
        self.assertEqual(ops.span_reads, 1)


class TestAPlantedZeroStandsOut(CustomTestCase):
    """Posten 1, can-fail half: 'not measured' and 'measured zero' must differ.

    This is the assertion that would have caught the original defect. The old
    fields were ``float = 0.0``; a phase line printed `0.000s` and read exactly
    like a device that did nothing. Two comments in phase_flip_boot.py were
    written on that reading.
    """

    def test_an_unmeasured_span_is_None_and_never_a_number(self):
        self.assertIsNone(RotationPhases().gpu_d2h_s)
        self.assertIsNone(RotationPhases().gpu_h2d_s)

    def test_the_report_says_not_measured_and_does_not_print_a_zero(self):
        # THE MUTANT THIS KILLS: change the defaults back to 0.0, or format
        # None as a float. Either way this line starts printing "0.000s".
        line = rotation_phase_report(RotationPhases(total_s=1.0, save_s=1.0))
        self.assertIn("gpu-span d2h not-measured / h2d not-measured", line)
        self.assertNotIn("gpu-span d2h 0.000s", line)

    def test_a_measured_zero_still_prints_as_a_number(self):
        # The other direction: the refusal must not swallow a real zero. A lane
        # that genuinely spanned no time is a finding, not an absence.
        line = rotation_phase_report(
            RotationPhases(total_s=1.0, save_s=1.0, gpu_d2h_s=0.0, gpu_h2d_s=2.5)
        )
        self.assertIn("gpu-span d2h 0.000s / h2d 2.500s", line)

    def test_the_cpu_path_reports_no_span_rather_than_a_zero_span(self):
        # Reachability: the REAL ops object on a device-free run must produce
        # the honest None, not a fabricated 0.0.
        phases = RotationPhases()
        _rotate(phases=phases)
        self.assertIsNone(phases.gpu_d2h_s)
        self.assertIsNone(phases.gpu_h2d_s)
        self.assertIn("not-measured", rotation_phase_report(phases))


class TestSpansBelongToOneLeg(CustomTestCase):
    """A reused ops object must not report a span reaching back to an older
    leg -- that number would look like a measurement and be a lifetime."""

    def test_reset_clears_the_markers(self):
        ops = TorchRotationOps()
        ops._span_first["d2h"] = object()
        ops._span_last["d2h"] = object()
        ops.reset_spans()
        self.assertIsNone(ops._span_first["d2h"])
        self.assertIsNone(ops._span_last["d2h"])

    def test_rotate_arena_resets_before_it_starts(self):
        # THE MUTANT THIS KILLS: drop the `ops.reset_spans()` call and a shared
        # ops object carries stale markers into the next leg.
        ops = TorchRotationOps()
        ops._span_first["h2d"] = "stale"
        _rotate(phases=RotationPhases(), ops=ops)
        self.assertNotEqual(ops._span_first["h2d"], "stale")


class TestTheCallTermsAreNamedForWhatTheyMeasure(CustomTestCase):
    """Posten 2: ``d2h_issue_s`` measured a completed transfer."""

    def test_the_issue_named_fields_are_gone(self):
        fields = set(RotationPhases.__dataclass_fields__)
        self.assertNotIn("d2h_issue_s", fields)
        self.assertNotIn("h2d_issue_s", fields)
        self.assertIn("d2h_call_s", fields)
        self.assertIn("h2d_call_s", fields)

    def test_the_report_and_the_dominant_label_both_moved(self):
        phases = RotationPhases(total_s=10.0, d2h_call_s=9.6, save_s=0.4)
        line = rotation_phase_report(phases)
        self.assertIn("d2h-call", line)
        self.assertNotIn("d2h-issue", line)
        self.assertEqual(phases.dominant()[0], "d2h_call")

    def test_both_call_terms_are_still_inside_accounted_s(self):
        # The rename must not drop a term out of the sum -- a phase set whose
        # parts stop adding up is the #846 class the reconciler exists for.
        # Asserted on the SUM directly rather than on residual_share, which is
        # size-dependent: a toy CPU rotation of ~50 KB is loop-overhead bound
        # and leaves ~17 % unaccounted, so a tolerance assertion here would be
        # measuring the fixture, not the rename.
        self.assertEqual(RotationPhases(d2h_call_s=3.0).accounted_s, 3.0)
        self.assertEqual(RotationPhases(h2d_call_s=2.0).accounted_s, 2.0)

    def test_the_call_term_is_actually_written_by_a_real_rotation(self):
        phases = RotationPhases()
        _rotate(phases=phases)
        self.assertGreater(phases.d2h_call_s, 0.0)
        self.assertGreater(phases.h2d_call_s, 0.0)


class TestOverlappedStepsCarriesNoExecutionInformation(CustomTestCase):
    """Posten 4: the counter is a closed form over (chunks, chunks, depth).

    IF THIS GOES RED BECAUSE THE COUNTER BECAME EXECUTION-DEPENDENT, THAT IS A
    FIX AND NOT A REGRESSION -- delete this class and restore the duplex claim
    in RotationStats. It is written as an assertion precisely so that such a
    change cannot happen silently while the old docstring is still believed.
    """

    SHAPES = (
        (INCOMING, OUTGOING, DEPTH),
        (OUTGOING, INCOMING, DEPTH),
        (INCOMING, INCOMING, DEPTH),
        (INCOMING, OUTGOING, 2),
        (INCOMING, OUTGOING, 1),
        (3 * CHUNK, 17 * CHUNK, DEPTH),
        (17 * CHUNK, 3 * CHUNK, DEPTH),
    )

    def test_the_counter_equals_the_closed_form_on_every_shape(self):
        for incoming, outgoing, depth in self.SHAPES:
            with self.subTest(incoming=incoming, outgoing=outgoing, depth=depth):
                stats = _rotate(depth=depth, incoming=incoming, outgoing=outgoing)
                h_chunks = -(-incoming // CHUNK)
                d_chunks = -(-outgoing // CHUNK)
                self.assertEqual(
                    stats.overlapped_steps,
                    plan_determined_overlap(h_chunks, d_chunks, depth),
                    "the counter moved away from the plan -- if that is "
                    "deliberate, the RotationStats retraction must be revisited",
                )

    def test_the_closed_form_needs_no_execution_at_all(self):
        # The point, stated without running anything: three numbers known
        # before the leg begins reproduce the counter.
        self.assertEqual(plan_determined_overlap(10, 12, 4), 11)
        self.assertEqual(plan_determined_overlap(12, 10, 4), 9)
        self.assertEqual(plan_determined_overlap(10, 10, 4), 9)
        self.assertEqual(plan_determined_overlap(3, 17, 4), 5)
        self.assertEqual(plan_determined_overlap(10, 12, 1), 0)
        self.assertEqual(plan_determined_overlap(0, 0, 4), 0)

    def test_at_production_depth_it_reduces_to_R_TIMEs_fitted_form(self):
        # R-TIME fitted `min(h, d) - 1 + [d > h]` and matched 12/12 legs. That
        # is this expression at depth 2, and depth 2 is the production default
        # (SGLANG_PHASE_FLIP_REFILL_DEPTH, environ.py:358). Asserted over a grid
        # rather than asserted in prose, because "the fit was a special case"
        # is the kind of claim that decays into "the fit was wrong".
        for h in range(0, 25):
            for d in range(0, 25):
                fitted = max(0, min(h, d) - 1 + (1 if d > h else 0))
                with self.subTest(h=h, d=d):
                    self.assertEqual(plan_determined_overlap(h, d, 2), fitted)

    def test_the_rendered_line_no_longer_calls_it_overlap(self):
        # A reader who greps the log must not find the word that started this.
        stats = RotationStats(steps=100, overlapped_steps=99)
        line = rotation_report("pp_to_tp", stats)
        self.assertIn("pipelined", line)
        self.assertIn("PLAN-DETERMINED", line)


if __name__ == "__main__":
    unittest.main()
