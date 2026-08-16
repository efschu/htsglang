"""#691: per-stage prefill compute/wait under PP, and a refusal when it cannot pair.

WHY THE TIMER WAS REFUSED UNDER PP. ``_install_rank_prefill_timer`` gated on
``pp_size != 1`` because ``RankPrefillLog`` pairs POSITIONALLY: ``flush()``
takes ``k = min(len(_pending), len(_durations))`` and pops k from both, so
index i of one queue is assumed to be index i of the other. If a stage ever
produces a timed forward with no matching record (or the reverse), that
assumption silently attaches one forward's duration to another forward's token
counts -- a wrong number reported with the same confidence as a right one.

WHAT THE EVIDENCE ACTUALLY SHOWS. On the reference PP=3 boot all three stages
emit the per-rank prefill line in near-equal counts (1020 / 1020 / 1018 over
one log rotation), i.e. every stage runs its own scheduling pass AND records
its own prefill batches. The two queues on a given stage are therefore fed by
that stage's own forwards, and per-stage local pairing is what the semantics
support -- which is also the only variant that delivers a per-stage
compute-vs-wait number. Last-stage-only reporting would measure one card.

WHAT THIS GUARD IS, STATED HONESTLY. It is a DIVERGENCE detector, not an
identity check. Per-side sequence numbers cannot catch an orphan: both sides
count their own stream, so an extra duration shifts the correspondence while
leaving the counters equal. An exact check needs a shared forward identity
threaded from the scheduler through ``ForwardBatch`` into the timer's metadata,
which touches the hot path and is named as the follow-up rather than smuggled
in here. What IS caught is the shape that breaks pairing in practice: one
stream persistently outrunning the other. The queues then stop being paired at
all, once, loudly, and the line degrades to the untimed form instead of
carrying a mispaired number.

A TRANSIENT LEAD IS LEGAL and must not trip it: under the overlap schedule a
forward's events routinely complete before the result that records them
(``test_duration_before_record_pairs_up`` in the sibling file pins that).
"""

import logging
import types
import unittest

import torch

torch.set_default_device("cpu")

from sglang.srt.managers.scheduler_components.metrics_reporter import (  # noqa: E402
    RankPrefillLog,
    SchedulerMetricsReporter,
)

LOGGER_NAME = "sglang.srt.managers.scheduler_components.metrics_reporter"


class FakeTimer:
    def __init__(self, log):
        self._log = log
        self.completed = []

    def _report(self):
        while self.completed:
            item = self.completed.pop(0)
            self._log._on_duration(item)


def _log_with_timer():
    log = RankPrefillLog()
    timer = FakeTimer(log)
    log.timer = timer
    return log, timer


class TheInstallGateAdmitsPipelines(unittest.TestCase):
    """The gate must stop refusing on pp_size > 1, and must keep refusing on a
    device that has no CUDA-alike events."""

    def _holder(self, *, device: str, pp_size: int):
        runner = types.SimpleNamespace(prefill_rank_timer=None)
        holder = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(
                device=device,
                server_args=types.SimpleNamespace(pp_size=pp_size),
                tp_worker=types.SimpleNamespace(model_runner=runner),
            ),
            rank_prefill_log=RankPrefillLog(),
        )
        holder._install_rank_prefill_timer = types.MethodType(
            SchedulerMetricsReporter._install_rank_prefill_timer, holder
        )
        return holder, runner

    def test_pp_three_now_installs_a_timer(self):
        holder, runner = self._holder(device="cuda", pp_size=3)
        holder._install_rank_prefill_timer()
        self.assertIsNotNone(
            runner.prefill_rank_timer,
            "PP still refuses the timer, so there is no per-stage "
            "compute-vs-wait number to arm",
        )
        self.assertIsNotNone(holder.rank_prefill_log.clock)

    def test_pp_one_is_unchanged(self):
        """Byte-neutrality of the existing install path."""
        holder, runner = self._holder(device="cuda", pp_size=1)
        holder._install_rank_prefill_timer()
        self.assertIsNotNone(runner.prefill_rank_timer)
        self.assertIsNotNone(holder.rank_prefill_log.clock)

    def test_a_non_cuda_device_is_still_refused(self):
        holder, runner = self._holder(device="cpu", pp_size=3)
        holder._install_rank_prefill_timer()
        self.assertIsNone(runner.prefill_rank_timer)
        self.assertIsNone(holder.rank_prefill_log.clock)


class ThePairingRefusesRatherThanMispairing(unittest.TestCase):
    """The planted mispairing: one stream outruns the other."""

    def test_a_persistent_orphan_stream_is_refused_loudly(self):
        log, timer = _log_with_timer()
        # Durations with no records at all: the PP shape the old gate feared.
        timer.completed.extend([0.01] * (RankPrefillLog.MAX_PAIR_SKEW + 4))
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            log.flush()
        self.assertTrue(any("pair" in line.lower() for line in cm.output), cm.output)
        self.assertTrue(log.pairing_refused)

    def test_after_refusal_the_line_degrades_instead_of_lying(self):
        log, timer = _log_with_timer()
        timer.completed.extend([0.01] * (RankPrefillLog.MAX_PAIR_SKEW + 4))
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING):
            log.flush()
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.record(new_tokens=8, cached_tokens=0, timed=True)
            log.flush()
        joined = "\n".join(cm.output)
        self.assertIn("#new-token: 8", joined)
        self.assertNotIn("gpu-ms", joined)

    def test_the_refusal_is_announced_once_not_per_flush(self):
        log, timer = _log_with_timer()
        timer.completed.extend([0.01] * (RankPrefillLog.MAX_PAIR_SKEW + 4))
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as first:
            log.flush()
        self.assertEqual(len(first.output), 1)
        # A second flush past the refusal must not re-announce.
        timer.completed.extend([0.01] * 4)
        with self.assertNoLogs(LOGGER_NAME, level=logging.WARNING):
            log.flush()

    def test_no_mispaired_line_is_emitted_at_the_moment_of_refusal(self):
        """The whole point: a wrong pairing must never reach the log."""
        log, timer = _log_with_timer()
        log.record(new_tokens=99, cached_tokens=0, timed=True)
        timer.completed.extend([0.01] * (RankPrefillLog.MAX_PAIR_SKEW + 4))
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.flush()
        # "gpu-ms:" with the colon, and only on the INFO line: the refusal
        # WARNING legitimately mentions gpu-ms in prose.
        timed_lines = [
            line for line in cm.output if line.startswith("INFO") and "gpu-ms:" in line
        ]
        self.assertEqual(
            timed_lines,
            [],
            "a timed line was emitted from queues that had already diverged",
        )


class TheHealthyPathIsUntouched(unittest.TestCase):
    """pp_size=1 behaviour, and any stage whose streams DO pair, must be
    byte-identical to before the guard existed."""

    def test_matched_pairing_still_emits_the_timed_line(self):
        log, timer = _log_with_timer()
        log.record(new_tokens=8, cached_tokens=2, timed=True)
        timer.completed.append(0.05)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.flush()
        self.assertEqual(len(cm.output), 1)
        self.assertIn("gpu-ms: 50.0", cm.output[0])
        self.assertFalse(log.pairing_refused)

    def test_a_transient_lead_does_not_trip_the_guard(self):
        """Overlap schedule: durations legitimately arrive before records."""
        log, timer = _log_with_timer()
        timer.completed.extend([0.01] * (RankPrefillLog.MAX_PAIR_SKEW - 1))
        log.flush()
        self.assertFalse(log.pairing_refused)
        for _ in range(RankPrefillLog.MAX_PAIR_SKEW - 1):
            log.record(new_tokens=4, cached_tokens=0, timed=True)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.flush()
        self.assertTrue(any("gpu-ms" in line for line in cm.output))
        self.assertFalse(log.pairing_refused)

    def test_records_leading_durations_also_do_not_trip_it(self):
        log, timer = _log_with_timer()
        for _ in range(RankPrefillLog.MAX_PAIR_SKEW - 1):
            log.record(new_tokens=4, cached_tokens=0, timed=True)
        log.flush()
        self.assertFalse(log.pairing_refused)


if __name__ == "__main__":
    unittest.main()
