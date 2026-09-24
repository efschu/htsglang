"""fnFL2 H20: FWD-TIMING-PREFILL, the component timeline of one prefill forward.

PP0 of the Next-Flash P group spends 5.5 s per 16384-token chunk; the three
island instruments (MOE-OFFLOAD-TIMING-PREFILL, ATTN-TIMING-PREFILL,
PLE-GATHER-PREFILL) name ~3.7 s of it and cannot say where the rest is, because
a sum of islands has no remainder term. The timeline records one event per
component boundary and names the segment that ends there; the cases pin the
properties the boot reading depends on:

* the Zaehlprobe: per-class sums add up to total_ms exactly (telescoping), so
  total_ms can be held against the rank's gpu-ms and other_ms is the only
  unnamed rest;
* switch off / not a plain prefill = no event is ever created (cost zero);
* every class the line prints has a writer in the source (a class without a
  mark would print 0.0 forever and read as "costs nothing");
* a raised forward leaves no half line behind.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import pathlib
import re
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers import fwd_timeline as ft
from sglang.test.test_utils import CustomTestCase


class _Ev:
    def __init__(self, t_ms: float):
        self.t = t_ms
        self.synced = False

    def elapsed_time(self, other: "_Ev") -> float:
        return other.t - self.t

    def synchronize(self) -> None:
        self.synced = True


class _Clock:
    """Hands out events whose stamps advance by the queued durations."""

    def __init__(self):
        self.t = 0.0
        self.next_dt = 0.0
        self.made = 0

    def event(self) -> _Ev:
        self.t += self.next_dt
        self.next_dt = 0.0
        self.made += 1
        return _Ev(self.t)


class _Mode:
    def __init__(self, plain: bool):
        self.plain = plain

    def is_plain_prefill(self) -> bool:
        return self.plain


_LINE = re.compile(r"FWD-TIMING-PREFILL forward=(\d+) tokens=(\d+) layers=(\d+) (.*)")


def _fields(line: str) -> dict:
    return {k: float(v) for k, v in re.findall(r"(\w+)_ms=([0-9.]+)", line)}


class TestFwdTimeline(CustomTestCase):
    def setUp(self):
        ft.reset_for_tests()
        # a case that leaves a forward open must not arm the marks of the
        # MoE tests that run after it in the same process (they have no GPU)
        self.addCleanup(ft.reset_for_tests)
        self.clock = _Clock()
        self.lines = []
        p = mock.patch.object(ft, "_new_event", self.clock.event)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(ft.logger, "info", self.lines.append)
        q.start()
        self.addCleanup(q.stop)

    def _seg(self, label: str, dt: float) -> None:
        self.clock.next_dt = dt
        ft.fwd_mark(label)

    def _one_layer_forward(self) -> dict:
        """A GDN layer with the expert-major MoE shape of PP0: two waves."""
        spent = {
            "embed": 3.0, "ple": 40.0 + 1100.0, "hc": 5.0 + 6.0 + 4.0,
            "dense": 10.0 + 7.0, "linear": 2.5, "shared": 1.5, "gate": 0.8,
            "moe_plan": 28.0 + 0.3, "moe_fetch": 5.4 + 5.1, "moe_apply": 1.4 + 1.3 + 0.9,
        }
        self._seg("embed", 3.0)
        self._seg("ple", 40.0)
        self._seg("ple", 1100.0)  # the pread gather, host-blocking
        self._seg("hc", 5.0)
        self._seg("dense", 10.0)
        self._seg("linear", 2.5)
        self._seg("dense", 7.0)
        self._seg("hc", 6.0)
        self._seg("shared", 1.5)
        self._seg("gate", 0.8)
        self._seg("moe_plan", 28.0)  # tolist rendezvous + planning
        self._seg("moe_fetch", 5.4)
        self._seg("moe_apply", 1.4)
        self._seg("moe_plan", 0.3)
        self._seg("moe_fetch", 5.1)
        self._seg("moe_apply", 1.3)
        self._seg("moe_apply", 0.9)
        self._seg("hc", 4.0)
        self.clock.next_dt = 2.0
        spent["other"] = 2.0
        return spent

    def test_segments_sum_to_total_and_land_in_their_class(self):
        with envs.SGLANG_WEG2_PREFILL_TIMING.override(True):
            self.assertTrue(ft.begin_if_timed(forward_mode=_Mode(True), tokens=16384, layers=29))
            spent = self._one_layer_forward()
            ft.fwd_end()
            self.assertEqual(self.lines, [], "read deferred to the next forward")
            ft.begin_if_timed(forward_mode=_Mode(True), tokens=16384, layers=29)
        self.assertEqual(len(self.lines), 1)
        m = _LINE.match(self.lines[0])
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1, 2, 3), ("1", "16384", "29"))
        f = _fields(self.lines[0])
        total = f.pop("total")
        self.assertAlmostEqual(sum(f.values()), total, places=0)
        self.assertAlmostEqual(total, sum(spent.values()), places=0)
        for label, ms in spent.items():
            self.assertAlmostEqual(f[label], ms, places=0, msg=label)

    def test_every_printed_class_is_in_the_line(self):
        ms = {label: 0.0 for label in ft.LABELS}
        ms["total"] = 0.0
        line = ft.format_line(forward=1, tokens=1, layers=1, ms=ms, marks=0)
        self.assertEqual(set(_fields(line)), set(ft.LABELS) | {"total"})

    def test_switch_off_creates_no_event(self):
        with envs.SGLANG_WEG2_PREFILL_TIMING.override(False):
            self.assertFalse(ft.begin_if_timed(forward_mode=_Mode(True), tokens=8, layers=1))
            self._seg("dense", 1.0)
            ft.fwd_end()
        self.assertEqual(self.clock.made, 0)
        self.assertEqual(self.lines, [])

    def test_decode_or_verify_is_not_timed(self):
        with envs.SGLANG_WEG2_PREFILL_TIMING.override(True):
            self.assertFalse(ft.begin_if_timed(forward_mode=_Mode(False), tokens=1, layers=29))
            self._seg("moe_plan", 1.0)
        self.assertEqual(self.clock.made, 0)

    def test_unknown_label_raises(self):
        with envs.SGLANG_WEG2_PREFILL_TIMING.override(True):
            ft.begin_if_timed(forward_mode=_Mode(True), tokens=8, layers=1)
            with self.assertRaises(ValueError):
                ft.fwd_mark("moe_fetchh")

    def test_aborted_forward_leaves_no_line(self):
        with envs.SGLANG_WEG2_PREFILL_TIMING.override(True):
            ft.begin_if_timed(forward_mode=_Mode(True), tokens=8, layers=1)
            self._seg("dense", 3.0)
            ft.fwd_abort()
            self._seg("dense", 3.0)  # no-op now
            ft.begin_if_timed(forward_mode=_Mode(True), tokens=8, layers=1)
        self.assertEqual(self.lines, [])

    def test_every_class_has_a_writer_in_the_source(self):
        """A class nobody marks prints 0.0 forever and reads as free."""
        root = pathlib.Path(ft.__file__).resolve().parents[1]
        sources = [
            root / "models" / "qwen4_exp.py",
            root / "models" / "qwen2_moe.py",
            root / "layers" / "moe" / "expert_offload.py",
            root / "layers" / "attention" / "hybrid_linear_attn_backend.py",
        ]
        text = "\n".join(p.read_text() for p in sources)
        written = set(re.findall(r'fwd_mark\("(\w+)"\)', text)) | {"other"}
        self.assertEqual(written, set(ft.LABELS))


if __name__ == "__main__":
    unittest.main()
