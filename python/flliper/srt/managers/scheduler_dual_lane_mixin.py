"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from flliper.srt.managers.schedule_batch import ScheduleBatch
from flliper.srt.observability.req_time_stats import set_time_batch
import logging
import time

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerDualLaneMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _pdflip_build_join_batch(self, reqs) -> None:
        """DUAL-TP3PP3 stage 2: prepare ``reqs`` exactly like an extend batch
        (slots, prefix rows, Mamba anchor, accounting, draft-cold arming), then
        convert them to decode-ready state without a forward; merged into the
        running batch at the next get_next_batch_to_run."""
        from flliper.srt.managers.phase_flip_draft_bootstrap import (
            arm_draft_cold_for_admission,
        )
        from flliper.srt.pdflip import dual_decode_join as _ddj

        set_time_batch(reqs, "set_forward_entry_time")
        jb = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        jb.prepare_for_extend()
        arm_draft_cold_for_admission(self, jb)
        _ddj.convert_to_joined(jb, self.future_map, self.enable_overlap)
        if not hasattr(self, "_pdflip_join_ready"):
            self._pdflip_join_ready = []
        self._pdflip_join_ready.append(jb)
        logger.info(
            "PDFLIP DECODE-JOIN n=%d rids=%s committed=%s: P-prefilled, joined the "
            "decode batch without an extend forward (the anchor token is computed "
            "by the next round)",
            len(reqs), [str(getattr(r, "rid", "?"))[:16] for r in reqs[:8]],
            [int(r.kv_committed_len) for r in reqs[:8]],
        )

    def _dual_group_lane_tick(self) -> None:
        """Multi-group runtime (#274): the two-class scheduler's grain
        boundary.  Rank-local by contract (lanes have no communicator); a
        no-op on every default path and on every rank without lanes.

        SERIAL mode (slice B): run one lane step inline. The serving group
        pays the whole step in wall time -- that is the +50 % price slice C
        exists to undercut.

        CONCURRENT mode (slice C): the lane runs on its own thread and its
        own high-priority stream, so this method never executes a forward.
        It is the point where the two CLASSES meet, and it does three things
        at the natural grain (an iteration boundary is a decode-step grain;
        chunked prefill hits it once per chunk):

        1. PROTECTED CLASS FIRST: if the lane has work that has not been
           submitted yet, yield briefly so the lane's kernels are enqueued
           before the scavenger's. Bounded by
           --dual-group-lane-admission-ms, measured per occurrence.
        2. SCAVENGER GETS THE IDLE BYTES: if the lane has been idle past the
           amortization threshold, lend it its configured segment.
        3. Never block on the lane's completion -- the serving group is
           work-conserving, so it goes on to build its own batch either way.
        """
        if not self.dual_group_lanes:
            return
        self._lane_share_sample()
        for lane in self.dual_group_lanes:
            if not lane.concurrent:
                if lane.has_work:
                    try:
                        lane.tick()
                    except Exception:
                        logger.exception(
                            "dual-group lane %d tick failed; dropping the active job.",
                            lane.lane_id,
                        )
                        # Releases the pool slots too: with one request slot
                        # per lane runner, dropping without freeing makes
                        # every later job fail in alloc_req_slots.
                        lane.drop_active()
                continue
            self._dual_group_admit(lane)

    def _lane_share_sample(self) -> None:
        """Feed one sample to the online card-equivalent estimator.

        Called at the same grain boundary as the two-class scheduler, which
        is the only place that sees BOTH classes' counters on one clock.

        The rung id is the controller state a window was measured under.
        Slice D1 builds no controller, so it is constant here -- the argument
        exists because DESIGN_201 addendum 12 (4) requires the bookkeeping to
        be in place BEFORE the first measurement, not retrofitted after one.
        """
        meter = self.lane_share_meter
        if meter is None:
            return
        # One float compare on the overwhelming majority of iterations. The
        # meter would return None anyway until its window is up, but BUILDING
        # the samples for it would not be free, and this call sits directly in
        # front of the serving group's batch launch -- everything spent here
        # is spent by the serving group (DESIGN_121 §12.7).
        now = time.perf_counter()
        if now < self._lane_share_next_t:
            return
        self._lane_share_next_t = now + meter.window_s * 0.25
        from flliper.srt.model_executor.lane_share import ClassSample

        samples = [
            ClassSample(
                "serving",
                {
                    "decode_tokens": self.metrics_reporter.gen_tokens_total,
                    "prefill_tokens": self.metrics_reporter.prefill_tokens_total,
                },
            )
        ]
        for lane in self.dual_group_lanes:
            # The device counters ride along with the work counters, from the
            # same instant: reading them from two different calls would put a
            # window boundary between the numerator and the denominator of the
            # occupancy this window reports (#284).
            clock = getattr(lane, "device_clock", None)
            samples.append(
                ClassSample(
                    f"lane{lane.lane_id}",
                    dict(lane.work_total),
                    device=None if clock is None else clock.snapshot().to_counters(),
                )
            )
        try:
            win = meter.observe(now, samples, rung=self._lane_rung())
        except Exception:
            logger.exception("lane share meter failed; disabling it for this boot.")
            self.lane_share_meter = None
            return
        if win is not None:
            self.metrics_reporter.log_lane_share(win)

    def _lane_rung(self) -> str:
        """Identity of the controller state the current window runs under."""
        return "static"

    def _dual_group_admit(self, lane) -> None:
        """One grain-boundary admission decision for a concurrent lane."""
        budget_ms = float(
            getattr(self.server_args, "dual_group_lane_admission_ms", 0.0) or 0.0
        )
        if lane.has_work and budget_ms > 0 and not lane._submitted.is_set():
            t0 = time.perf_counter()
            # Waiting on the event RELEASES the GIL, which is the actual
            # mechanism: the lane thread needs the interpreter to issue its
            # launches, and the scheduler thread holding it is the only way
            # the scavenger could starve the protected class.
            lane._submitted.wait(timeout=budget_ms / 1000.0)
            lane.note_admission((time.perf_counter() - t0) * 1000.0)
        if lane.lending is not None:
            lane.lending.maybe_lend()
