"""(g) The GGUF STEP-0 phase lines carry the loading THREAD's CPU time and
faults, so a boot can tell starvation from work.

WHY.  weg2rc5gg / weg2rc6gg: group D's GGUF producer took 338-715 s per rank
against 60-72 s on group P for the SAME code (gguf_quant_weights_iterator +
Qwen35GGUFAdapter.transform_stream), while the transform's own numpy work
measured 0.09-0.22 s per ssm_out tensor on a quiet box (wall == CPU) and
1-35 s per tensor while the host ran its Docker build / acceptance.  The
STEP-0 lines only carried WALL time, so the boot could not say whether the
loading thread computed (cpu/wall ~ 1) or waited -- runnable but not running
(involuntary switches), or blocked on disk (major faults, voluntary switches).
"""

from unittest import mock

from sglang.srt.model_loader import loader as L
from sglang.test.test_utils import CustomTestCase


def _fake_clock(readings):
    """A rusage source that hands out ``readings`` in order."""
    it = iter(readings)
    return lambda: next(it)


def _r(user, sys_, minflt, majflt, nvcsw=0, nivcsw=0):
    return L.Step0Rusage(user, sys_, minflt, majflt, nvcsw, nivcsw)


class TheNoteNamesCpuFaultsAndSwitches(CustomTestCase):
    def test_a_computing_phase_reads_cpu_per_wall_one(self):
        """RED on e714c95546: no meter existed."""
        note = L.step0_cpu_note(10.0, _r(9.0, 1.0, 2000, 0, 3, 5))
        self.assertIn("cpu[thread] user=9.00s sys=1.00s", note)
        self.assertIn("minflt=2000 majflt=0", note)
        self.assertIn("vcsw=3 ivcsw=5", note)
        self.assertIn("cpu/wall=1.00", note)

    def test_a_starved_phase_reads_cpu_per_wall_far_below_one(self):
        """weg2rc6gg's D shape if it was starvation: 700 s wall, ~70 s CPU,
        no disk faults, many involuntary switches."""
        note = L.step0_cpu_note(700.0, _r(60.0, 10.0, 5_000_000, 12, 900, 250_000))
        self.assertIn("cpu/wall=0.10", note)
        self.assertIn("majflt=12", note)
        self.assertIn("ivcsw=250000", note)

    def test_zero_wall_does_not_divide(self):
        self.assertIn("cpu/wall=n/a", L.step0_cpu_note(0.0, _r(0, 0, 0, 0)))

    def test_the_delta_is_per_field(self):
        d = L.Step0Rusage.delta(_r(1.0, 2.0, 10, 1, 4, 5), _r(3.5, 2.25, 25, 3, 6, 9))
        self.assertEqual(tuple(d), (2.5, 0.25, 15, 2, 2, 4))


class TheProducerMeterCountsOnlyTheProducer(CustomTestCase):
    def test_rusage_is_accumulated_around_next_only(self):
        """The consumer's work between two next() calls is NOT the producer's:
        only the deltas across each next() accumulate."""
        readings = [
            _r(0.0, 0.0, 0, 0), _r(1.0, 0.5, 10, 1),     # next() #1: +1.0/+0.5
            _r(5.0, 0.5, 10, 1), _r(6.0, 1.0, 30, 1),     # consumer ran 4 s; next() #2
            _r(9.0, 1.0, 30, 1), _r(9.5, 1.0, 31, 2),     # final next() -> StopIteration
        ]
        meter = L.Step0ProducerMeter(rusage=_fake_clock(readings))
        got = list(meter.wrap(iter(["a", "b"])))
        self.assertEqual(got, ["a", "b"])
        self.assertEqual(meter.n, 2)
        # 1.0 + 1.0 + 0.5 user (the closing next() is the producer's tail too,
        # as the wall split always counted it); the consumer's 4 s + 3 s are out
        self.assertEqual(tuple(meter.cpu), (2.5, 1.0, 31, 2, 0, 0))

    def test_wall_time_is_kept_as_before(self):
        clock = iter([0.0, 0.5, 10.0, 10.25, 20.0, 20.125])
        meter = L.Step0ProducerMeter(rusage=lambda: _r(0, 0, 0, 0))
        with mock.patch.object(L.time, "perf_counter", lambda: next(clock)):
            list(meter.wrap(iter([1, 2])))
        self.assertAlmostEqual(meter.t, 0.5 + 0.25 + 0.125)


class TheLoaderUsesTheMeter(CustomTestCase):
    def test_every_step0_line_of_the_gguf_loader_carries_the_cpu_note(self):
        import inspect

        src = inspect.getsource(L.GGUFModelLoader.load_model)
        for phase in ("gguf name-map", "structure alloc", "load_weights (rank",
                      "process_weights_after_loading (flat-assembly", "SUMMARY rank"):
            i = src.index(phase)
            window = src[i:i + 900]
            self.assertIn("step0_cpu_note", window, f"{phase!r} line carries no cpu note")
        self.assertIn("Step0ProducerMeter", src)

    def test_the_thread_is_measured_not_the_process(self):
        import resource

        want = getattr(resource, "RUSAGE_THREAD", resource.RUSAGE_SELF)
        self.assertEqual(L._STEP0_RUSAGE_WHO, want)
        self.assertEqual(len(L.step0_rusage_now()), 6)
