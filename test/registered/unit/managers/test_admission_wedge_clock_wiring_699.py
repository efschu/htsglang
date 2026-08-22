"""#699: wire the admission-wedge detector to a real first-token-progress clock.

The detector itself (``admission_wedge_verdict``, landed in 9c686ca936) is a
pure function of ``(queued, running, seconds_since_progress)`` and was already
proven correct against the specimen shape. What it did NOT have was a real
clock: something in the serving path that records, per request, the moment
its FIRST output token is produced, so "seconds_since_progress" can be
computed from a live scheduler instead of a hand-fed float.

THE BLIND SPOT THIS FILE GUARDS AGAINST. The #699 commit message is explicit
that forward_ct is the WRONG clock: chunked prefill advances forward_ct for
tens of seconds while zero requests reach a first token. A wiring that
(accidentally or by "simplification") stamps the progress clock at forward
time instead of first-token-commit time would silently resurrect exactly the
blindness #699 exists to fix -- and would still pass every test in
test_admission_wedge_detector_699.py, because that file only tests the pure
function, never the wiring. This file is the wiring's own regression net.

Two things are tested hermetically (CUDA_VISIBLE_DEVICES=""), no scheduler
boot, no CUDA:

1. ``check_admission_wedge_once`` (invariant_checker.py): reads queued/
   running/age off a duck-typed fake scheduler object and reproduces the
   31.64 s specimen end to end, including staying silent while the box is
   actually serving.

2. ``SchedulerBatchResultProcessor.process_batch_result_prefill``
   (batch_result_processor.py): the real serving-path method that appends a
   request's first output token. A request whose prefill chunk FINISHES this
   round must call the progress-clock callback exactly once; a request whose
   chunk is still IN FLIGHT (a forward pass ran, but no first token was
   produced) must NOT call it. That second assertion is the mutation guard:
   a clock wired to "a forward pass happened" instead of "a first token was
   committed" fails it while every existing test stays green.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    ADMISSION_WEDGE_SECONDS,
    check_admission_wedge_once,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --- part 1: the polling function that reads scheduler state -------------


class _FakeRunningBatch:
    def __init__(self, reqs):
        self.reqs = reqs


class _FakeScheduler:
    """Duck-typed stand-in for Scheduler: only the attributes the wedge
    watchdog reads (#699 wiring scope), never a real scheduler boot."""

    def __init__(self, waiting_queue, running_reqs, last_first_token_progress_time):
        self.is_initializing = False
        self.waiting_queue = waiting_queue
        self.running_batch = _FakeRunningBatch(running_reqs)
        self.last_first_token_progress_time = last_first_token_progress_time


class TestCheckAdmissionWedgeOnce(CustomTestCase):
    def test_the_31_second_specimen_alarms_through_the_scheduler_reading(self):
        now = time.perf_counter()
        sched = _FakeScheduler(
            waiting_queue=[object()],
            running_reqs=[],
            last_first_token_progress_time=now - 31.64,
        )
        alarm, detail = check_admission_wedge_once(sched, now=now)
        self.assertTrue(alarm, detail)
        self.assertIn("ADMISSION-WEDGE", detail)

    def test_busy_scheduler_is_silent_regardless_of_clock_age(self):
        """The box is serving (1 running): stays silent even if the progress
        clock is stale, e.g. a long decode with no NEW first token lately."""
        now = time.perf_counter()
        sched = _FakeScheduler(
            waiting_queue=[object(), object()],
            running_reqs=[object()],
            last_first_token_progress_time=now - 120.0,
        )
        alarm, detail = check_admission_wedge_once(sched, now=now)
        self.assertFalse(alarm, detail)

    def test_below_threshold_since_last_progress_is_silent(self):
        now = time.perf_counter()
        sched = _FakeScheduler(
            waiting_queue=[object()],
            running_reqs=[],
            last_first_token_progress_time=now - (ADMISSION_WEDGE_SECONDS - 1.0),
        )
        alarm, _ = check_admission_wedge_once(sched, now=now)
        self.assertFalse(alarm)

    def test_initializing_scheduler_is_never_polled(self):
        now = time.perf_counter()
        sched = _FakeScheduler(
            waiting_queue=[object()],
            running_reqs=[],
            last_first_token_progress_time=now - 999.0,
        )
        sched.is_initializing = True
        alarm, detail = check_admission_wedge_once(sched, now=now)
        self.assertFalse(alarm)
        self.assertIn("initializing", detail)

    @patch("sglang.srt.managers.scheduler_components.invariant_checker.logger")
    def test_a_firing_alarm_logs_a_single_loud_line_naming_age_and_clock(
        self, mock_logger
    ):
        now = time.perf_counter()
        last_progress = now - 31.64
        sched = _FakeScheduler(
            waiting_queue=[object()],
            running_reqs=[],
            last_first_token_progress_time=last_progress,
        )
        alarm, detail = check_admission_wedge_once(sched, now=now, log_on_alarm=True)
        self.assertTrue(alarm)
        mock_logger.error.assert_called_once()
        (logged,), _ = mock_logger.error.call_args
        self.assertIn("ADMISSION-WEDGE", logged)
        self.assertIn("31.6", logged)
        self.assertIn(f"{last_progress:.1f}", logged)


# --- part 2: the real first-token commit point in the serving path -------


def _make_processor(record_first_token_progress) -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False, enable_hisparse=False),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(report_prefill_stats=lambda **k: None),
        draft_worker=None,
        model_worker=SimpleNamespace(),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(stream_output=lambda *a, **k: None),
        abort_request=lambda *a, **k: None,
        record_first_token_progress=record_first_token_progress,
    )


def _make_req(rid: str) -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        vocab_size=32000,
    )
    return req


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.return_logprob = False
        self.decoding_reqs = []
        self.prefill_stats = None
        self.dp_cooperation_info = None


class _FakeGenerationResult:
    def __init__(self, next_token_ids):
        self.copy_done = None
        self.routed_experts_output = None
        self.indexer_topk_output = None
        self.logits_output = None
        self.next_token_ids = torch.tensor(next_token_ids, dtype=torch.long)
        self.extend_input_len_per_req = None
        self.extend_logprob_start_len_per_req = None
        self.can_run_cuda_graph = False


class TestFirstTokenCommitCallsTheProgressClock(CustomTestCase):
    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor."
        "maybe_cache_unfinished_req"
    )
    def test_a_finishing_prefill_chunk_records_progress_exactly_once(self, _mock_cache):
        """req.inflight_middle_chunks <= 0: this round genuinely appends the
        FIRST output token (output_ids was empty going in). This is the
        first-token-commit instant the clock must stamp."""
        req = _make_req("r0")
        self.assertEqual(list(req.output_ids), [])
        calls = []
        proc = _make_processor(record_first_token_progress=lambda: calls.append(1))
        batch = _FakeBatch([req])
        result = _FakeGenerationResult(next_token_ids=[999])

        proc.process_batch_result_prefill(batch, result)

        self.assertEqual(list(req.output_ids), [999], "first token was not committed")
        self.assertEqual(
            len(calls),
            1,
            "progress clock must be stamped exactly once when a request's "
            "first output token is committed",
        )

    def test_a_continuing_chunked_prefill_does_NOT_record_progress(self):
        """CAN-FAIL / mutation guard. req.inflight_middle_chunks > 0: a
        forward pass ran (chunked prefill continues) but NO output token was
        produced -- output_ids stays empty. A clock wired to forward-time
        instead of first-token-time would fire here; this is exactly the
        blindness #699 exists to avoid (forward_ct advances, nothing
        progresses)."""
        req = _make_req("r1")
        req.inflight_middle_chunks = 1
        calls = []
        proc = _make_processor(record_first_token_progress=lambda: calls.append(1))
        batch = _FakeBatch([req])
        result = _FakeGenerationResult(next_token_ids=[999])

        proc.process_batch_result_prefill(batch, result)

        self.assertEqual(
            list(req.output_ids),
            [],
            "a continuing chunk must not emit an output token",
        )
        self.assertEqual(
            len(calls),
            0,
            "progress clock must NOT be stamped when no first token was "
            "produced -- this is the forward_ct-shaped blindness #699 exists "
            "to catch",
        )


if __name__ == "__main__":
    unittest.main()
