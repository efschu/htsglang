"""fnFL2 H67: the prefill timing flush must not join the pending PP send.

Measured (x166/x167, PP=3 P group 29/11/8 layers, 97k request in 16k chunks):
PP0 (5090) spends +548/+639 ms in chunks 4/5 inside its FWD-TIMING 'linear'
segment and +1196 ms in burst forward 16, while ATTN-TIMING (the event pair
AROUND forward_extend) shows linear 77.5 / 77.5 / 24.8 ms. The only thing
between the two instruments at the stage's head layer is the ATTN-TIMING
flush, and it was ``torch.cuda.synchronize()``: a wait for every stream of the
process. On a non-last stage one of those streams carries the async proxy send
of the previous chunk (NCCL isend, posted and never joined on the pass path),
which completes only when the next stage posts its receive -- after PP1 (3.8 s
per chunk) has finished its own previous chunk. PP0 (3.16 s per chunk since
H32) therefore sat inside its forward for PP1's remainder; every PP0 chunk 4..6
ends 3.25-3.40 s after PP1's chunk k-2, and the rank's gpu-ms booked PP1's
pace as PP0 compute.

The cases pin the fix (SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH):

* switch on: the ATTN-TIMING, MOE-OFFLOAD-TIMING-PREFILL and token-major
  MOE-OFFLOAD-TIMING flushes wait on the flushed events only -- never
  ``torch.cuda.synchronize()`` -- and report the same sums;
* a device whose only unfinished work is a foreign stream (the pending send)
  costs the flush that stream's remaining time in the legacy mode and nothing
  in the event mode (TIMING-FLUSH-WAIT wait_ms);
* an event recorded on another stream that is not complete yet is still
  waited for (the flush never reads an unfinished event);
* switch off: the old device-wide wait, byte-for-byte the old form.

On the pre-fix tree the helper, the switch and the TIMING-FLUSH-WAIT line do
not exist: every case fails.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import types
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers import prefill_timing as pt
from sglang.test.test_utils import CustomTestCase


class _Ev:
    """A CUDA event stand-in on a fake stream: complete once its stream got
    that far (``done``), ``synchronize`` completes it and counts the wait."""

    def __init__(self, t_ms: float, done: bool = True):
        self.t = t_ms
        self.done = done
        self.synced = 0

    def elapsed_time(self, other: "_Ev") -> float:
        if not (self.done and other.done):
            raise RuntimeError("cudaErrorNotReady: elapsed_time on an unfinished event")
        return other.t - self.t

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.synced += 1
        self.done = True


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class _Clock:
    """time.monotonic stand-in: advances only when the fake device sync runs."""

    def __init__(self):
        self.t = 100.0

    def monotonic(self) -> float:
        return self.t


def _no_device_sync():
    raise AssertionError(
        "torch.cuda.synchronize() in the flush: a device-wide wait joins the "
        "pending PP proxy send of the previous chunk"
    )


class _LogCase(CustomTestCase):
    loggers = ()

    def setUp(self):
        self.cap = _Capture()
        self._saved = []
        for name in ("sglang.srt.layers.prefill_timing",) + tuple(self.loggers):
            log = logging.getLogger(name)
            self._saved.append((log, log.level))
            log.addHandler(self.cap)
            log.setLevel(logging.INFO)

    def tearDown(self):
        for log, level in self._saved:
            log.removeHandler(self.cap)
            log.setLevel(level)

    def lines(self, prefix):
        return [l for l in self.cap.lines if l.startswith(prefix)]


class TestFlushWait(_LogCase):
    def test_event_mode_waits_on_the_events_not_on_the_device(self):
        evs = [_Ev(0.0), _Ev(3.0), _Ev(4.0, done=False)]
        with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(True), mock.patch(
            "torch.cuda.synchronize", side_effect=_no_device_sync
        ):
            pt.flush_wait(evs, instrument="attn", forward=4)
        self.assertTrue(all(e.done for e in evs))
        self.assertEqual(evs[-1].synced, 1)  # the last recorded one is waited for
        (line,) = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("instrument=attn forward=4 mode=event", line)
        self.assertIn("events=3", line)

    def test_an_unfinished_event_on_another_stream_is_still_waited_for(self):
        # the last one is complete, an earlier one (another stream) is not
        evs = [_Ev(0.0), _Ev(1.0, done=False), _Ev(2.0)]
        with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(True), mock.patch(
            "torch.cuda.synchronize", side_effect=_no_device_sync
        ):
            pt.flush_wait(evs, instrument="moe_prefill", forward=2)
        self.assertEqual(evs[1].synced, 1)
        self.assertEqual(evs[0].synced, 0)  # complete ones are only queried
        self.assertEqual(evs[0].elapsed_time(evs[1]), 1.0)

    def test_pending_pp_send_costs_the_device_flush_and_not_the_event_flush(self):
        """The mechanism: every flushed event is done; the device still owes
        600 ms on a foreign stream (the proxy send that waits for the next
        stage's receive). The legacy flush pays them inside the forward."""
        clock = _Clock()

        def _device_sync():
            clock.t += 0.600  # the send completes when the next stage receives

        evs = [_Ev(0.0), _Ev(2.0)]
        fake_time = types.SimpleNamespace(monotonic=clock.monotonic, time=lambda: 1790283001.846)
        with mock.patch.object(pt, "time", fake_time), mock.patch(
            "torch.cuda.synchronize", side_effect=_device_sync
        ) as dev:
            with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(False):
                legacy = pt.flush_wait(evs, instrument="attn", forward=4)
            with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(True):
                event = pt.flush_wait(evs, instrument="attn", forward=5)
        self.assertEqual(dev.call_count, 1)
        self.assertAlmostEqual(legacy, 600.0, places=3)
        self.assertEqual(event, 0.0)
        a, b = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("forward=4 mode=device wait_ms=600.0", a)
        self.assertIn("forward=5 mode=event wait_ms=0.0", b)
        # the return time names WHEN the wait ended (held against the next
        # stage's WEG2-VRAM-PEAK t_unix_ms of its chunk k-2 on the metal)
        self.assertIn("t_unix_ms=1790283001846", a)

    def test_default_is_the_old_device_wide_wait(self):
        self.assertFalse(envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.get())
        evs = [_Ev(0.0), _Ev(1.0)]
        with mock.patch("torch.cuda.synchronize") as dev:
            pt.flush_wait(evs, instrument="attn", forward=1)
        self.assertEqual(dev.call_count, 1)
        self.assertEqual([e.synced for e in evs], [0, 0])
        (line,) = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("mode=device", line)


class TestAttnTimingFlush(_LogCase):
    loggers = ("sglang.srt.layers.attention.hybrid_linear_attn_backend",)

    def _two_forwards(self):
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hb

        hb._ATTN_T.update(
            {"on": None, "ev": {"full": [], "linear": []}, "layers": 0, "tokens": 0, "forwards": 0}
        )
        hb._ATTN_T_HEAD.head = None
        first = []
        try:
            # stage 0 of the 29/11/8 split, 4 layers shown: GDN 0-2, full 3
            for fwd in range(2):
                for layer in range(4):
                    hb._attn_timing_begin(layer, 16384)
                    if fwd == 1:
                        break  # the head layer of forward 2 flushes forward 1
                    kind = "full" if layer == 3 else "linear"
                    e0, e1 = _Ev(0.0), _Ev(5.0 if kind == "full" else 2.0)
                    first += [e0, e1]
                    hb._attn_timing_note(kind, e0, e1)
        finally:
            hb._ATTN_T_HEAD.head = None
        return first

    def test_event_mode_flush_at_the_head_layer(self):
        with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(True), mock.patch(
            "torch.cuda.synchronize", side_effect=_no_device_sync
        ):
            first = self._two_forwards()
        (line,) = self.lines("ATTN-TIMING-PREFILL")
        self.assertIn("forward=1 tokens=16384 layers=4 full_attn_ms=5.0 (1 layers)", line)
        self.assertIn("linear_attn_ms=6.0 (3 layers)", line)
        self.assertTrue(all(e.done for e in first))
        (wait,) = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("instrument=attn forward=1 mode=event", wait)
        self.assertIn("events=8", wait)

    def test_legacy_flush_keeps_one_device_sync_per_forward(self):
        with mock.patch("torch.cuda.synchronize") as dev:
            self._two_forwards()
        self.assertEqual(dev.call_count, 1)
        (line,) = self.lines("ATTN-TIMING-PREFILL")
        self.assertIn("full_attn_ms=5.0 (1 layers) linear_attn_ms=6.0 (3 layers)", line)
        (wait,) = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("mode=device", wait)


class TestMoeTimingFlush(_LogCase):
    loggers = ("sglang.srt.layers.moe.expert_offload",)

    def test_prefill_flush_event_mode_on_a_stage_starting_at_layer_29(self):
        from sglang.srt.layers.moe import expert_offload as eo

        eo._WAVE_TP.update({"ev": [], "layers": 0, "waves": 0, "spill": 0, "tokens": 0, "forwards": 0})
        eo._WAVE_TP_HEAD.head = None
        first = []
        try:
            with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(True), mock.patch(
                "torch.cuda.synchronize", side_effect=_no_device_sync
            ):
                for fwd in range(2):
                    for layer in range(29, 40):
                        ev = (_Ev(0.0), _Ev(3.0), _Ev(4.0))
                        if fwd == 0:
                            first += list(ev)
                        eo._wave_timing_note_prefill(layer, [ev], 5, 4096)
        finally:
            eo._WAVE_TP_HEAD.head = None
        (line,) = self.lines("MOE-OFFLOAD-TIMING-PREFILL")
        self.assertIn("tokens=4096 layers=11 waves=11 spill_experts=55", line)
        self.assertIn("fetch_ms=33.0 apply_ms=11.0", line)
        self.assertTrue(all(e.done for e in first))
        (wait,) = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("instrument=moe_prefill forward=1 mode=event", wait)
        self.assertIn("events=33", wait)

    def test_token_major_flush_event_mode(self):
        from sglang.srt.layers.moe import expert_offload as eo

        eo._WAVE_T.update({"ev": [], "fetch_ms": 0.0, "apply_ms": 0.0, "layers": 0, "spill": 0,
                           "tokens": 0, "forwards0": 0})
        eo._WAVE_T_HEAD.head = None
        try:
            with envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.override(True), mock.patch(
                "torch.cuda.synchronize", side_effect=_no_device_sync
            ):
                for fwd in range(16):
                    for layer in (29, 30):
                        eo._wave_timing_note(layer, _Ev(0.0), _Ev(1.0), _Ev(3.0), 1, 4)
        finally:
            eo._WAVE_T_HEAD.head = None
        (line,) = self.lines("MOE-OFFLOAD-TIMING forwards=16")
        self.assertIn("fetch_ms=31.0 apply_ms=62.0", line)
        (wait,) = self.lines("TIMING-FLUSH-WAIT")
        self.assertIn("instrument=moe forward=16 mode=event", wait)


if __name__ == "__main__":
    unittest.main()
