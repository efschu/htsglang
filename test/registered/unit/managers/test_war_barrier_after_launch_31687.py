"""upstream #31687: the WAR barrier sits right after each run_batch launch.

Hermetic, CPU. Drives the REAL ``Scheduler.event_loop_overlap`` (and the
fork's concurrent spill lane ``_dispatch_concurrent_spill``) with a recording
fake for three iterations and pins the ORDER of events:

* every ``run_batch`` launch is followed by ``_apply_war_barrier`` BEFORE the
  previous batch's result is processed -- before the fix the barrier ran at
  the loop head, so the result processing of batch k-1 (which runs after
  batch k's launch) was not ordered behind batch k's shared-buffer reads;
* an idle iteration (no batch) applies no barrier of its own;
* the spill lane fences its own forward inside the stream swap, i.e. while
  ``forward_stream`` still names the spill stream, and after the device
  forward's barrier (so the device read-done event is consumed first);
* the disagg overlap loops carry the same placement (source check; the PD
  disaggregation loops are not reachable on weg2 but mirror upstream).
"""

import inspect
from collections import deque
from types import SimpleNamespace

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers.scheduler import Scheduler


class _Batch:
    def __init__(self, name):
        self.name = name
        self.forward_mode = SimpleNamespace(name="DECODE")

    def copy(self):
        return self


class _Fake:
    """Just enough of a Scheduler for event_loop_overlap."""

    def __init__(self, plan):
        self.log = []
        self._plan = deque(plan)
        self._iters = 0
        self._engine_paused = False
        self.request_receiver = SimpleNamespace(recv_requests=lambda: [])
        self.running_batch = None
        self.last_batch = None
        self.idle_sleeper = None
        self._pending_spill_batch = None
        self.is_generation = True
        self.enable_unified_memory = False
        self.forward_stream = "device_stream"

    @property
    def gracefully_exit(self):
        return not self._plan

    def process_input_requests(self, reqs):
        pass

    def _dual_group_lane_tick(self):
        pass

    def get_next_batch_to_run(self, running_batch, last_batch):
        b = self._plan.popleft()
        return SimpleNamespace(running_batch=running_batch, batch_to_run=b)

    def is_disable_overlap_for_batch(self, batch, last_batch):
        return False

    def run_batch(self, batch):
        self.log.append(("run", batch.name, self.forward_stream))
        return SimpleNamespace(name=batch.name)

    def _apply_war_barrier(self):
        self.log.append(("barrier", self.forward_stream))

    def process_batch_result(self, batch, result):
        self.log.append(("process", batch.name))

    def _weg2_post_wake_pass_log(self, batch):
        pass

    def on_idle(self):
        self.log.append(("idle",))

    def launch_batch_sample_if_needed(self, result, batch):
        pass


def _drive(monkeypatch, plan):
    fake = _Fake(plan)
    monkeypatch.setattr(sched_mod, "_stage_sync", lambda *a, **k: None)
    Scheduler.event_loop_overlap(fake)
    return fake.log


def test_barrier_follows_every_launch_before_prior_result_is_processed(monkeypatch):
    log = _drive(monkeypatch, [_Batch("b1"), _Batch("b2"), _Batch("b3")])
    assert log == [
        ("run", "b1", "device_stream"),
        ("barrier", "device_stream"),
        ("run", "b2", "device_stream"),
        ("barrier", "device_stream"),
        ("process", "b1"),
        ("run", "b3", "device_stream"),
        ("barrier", "device_stream"),
        ("process", "b2"),
    ]


def test_idle_iteration_applies_no_barrier_of_its_own(monkeypatch):
    log = _drive(monkeypatch, [_Batch("b1"), None])
    assert log == [
        ("run", "b1", "device_stream"),
        ("barrier", "device_stream"),
        ("process", "b1"),
    ]


def test_spill_lane_fences_its_forward_inside_the_swap():
    fake = _Fake([])
    fake.spill_stream = "spill_stream"
    fake.forward_stream_ctx = "device_ctx"
    fake.spill_stream_ctx = "spill_ctx"
    fake.enable_overlap = True
    fake.batch_record_buf, fake.batch_record_ct = [None, None], 0
    fake._spill_record_buf, fake._spill_record_ct = [None, None], 0
    fake._spill_result_queue = deque()
    # the device lane launched first and fenced itself
    fake.log.extend([("run", "dev", "device_stream"), ("barrier", "device_stream")])
    Scheduler._dispatch_concurrent_spill(fake, _Batch("spill1"))
    assert fake.log[2:] == [
        ("run", "spill1", "spill_stream"),
        ("barrier", "spill_stream"),
    ]
    assert fake.forward_stream == "device_stream"  # swap restored


def test_disagg_loops_launch_then_fence():
    from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
    from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    for fn in (
        SchedulerDisaggregationPrefillMixin.event_loop_overlap_disagg_prefill,
        SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode,
        Scheduler.event_loop_overlap,
    ):
        src = inspect.getsource(fn)
        assert src.count("self._apply_war_barrier()") == 1, fn.__name__
        launch = src.index("batch_result = self.run_batch(batch)")
        barrier = src.index("self._apply_war_barrier()")
        plan = src.index("plan = self.get_next")
        assert plan < launch < barrier, fn.__name__
