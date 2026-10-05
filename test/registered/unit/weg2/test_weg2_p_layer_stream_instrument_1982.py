# SPDX-License-Identifier: Apache-2.0
"""deskq 1988 (c) / report 1982 V0: ms fields in the P-LAYER-STREAM OUT / REGAIN lines (pure log). The instrument
changes no behaviour (outputs, counters, freed bytes) and carries every field.
"""
from __future__ import annotations

import os
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.weg2 import p_layer_stream as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _toy(n=4, width=8):
    torch.manual_seed(0)
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Linear(width, width) for _ in range(n)])
    return m


class _Saver:
    def __init__(self, units):
        self.units = {u.tag: u for u in units}
        self.saved = {}

    def pause(self, tag):
        self.saved[tag] = [(t, t.data.clone()) for ts in self.units[tag].tensors.values() for t in ts]
        for t, _c in self.saved[tag]:
            t.data.fill_(1e6)

    def resume(self, tag):
        for t, c in self.saved.pop(tag):
            t.data.copy_(c)


class StreamInstrument1982(CustomTestCase):
    def setUp(self):
        L.reset_for_tests()

    def tearDown(self):
        L.reset_for_tests()

    def _run(self):
        m = _toy()
        units = [L.StreamUnit("weights_1", 1000, {i: [m.layers[i].weight, m.layers[i].bias] for i in (1, 2)})]
        sv = _Saver(units)
        st = L.LayerStreamer(units, pause=sv.pause, resume=sv.resume, prefetch=1, device=None, pin=False)
        st.install_hooks(m.layers, 0, len(m.layers))
        lines = []
        with mock.patch.object(L.logger, "warning", lambda msg, *a: lines.append(msg % a)):
            freed = st.stream_out(["weights_1"])
            back = st.regain("weights_1")
        return st, freed, back, lines

    def test_out_and_regain_lines_carry_every_ms_field(self):
        _st, freed, back, lines = self._run()
        out = [x for x in lines if " OUT tag=" in x]
        reg = [x for x in lines if " REGAIN tag=" in x]
        self.assertEqual((len(out), len(reg)), (1, 1))
        for f in ("sync_pre=", "pin=", "copy=", "images=", "sync_post=", "tms_pause=", "out_total="):
            self.assertIn(f, out[0])
        for f in ("release_ring=", "tms_resume=", "h2d_sync=", "regain_total=", "out_to_regain_s="):
            self.assertIn(f, reg[0])
        self.assertNotIn("out_to_regain_s=n/a", reg[0], "the OUT stamp reaches the REGAIN line")
        self.assertIn("freed=1000 B", out[0], "the old prefix of the line is intact")
        self.assertIn("resident again (graphs valid: same VA)", reg[0])

    def test_behaviour_unchanged(self):
        st, freed, back, _lines = self._run()
        self.assertEqual((freed, back), (1000, 1000))
        self.assertEqual(st.counters["out"], 1)
        self.assertEqual(st.counters["regain"], 1)
        self.assertEqual(st.paused(), ())
        self.assertEqual(st._out_t, {}, "the stamp is dropped with the regain")

    def test_regain_without_a_stamp_says_na(self):
        m = _toy()
        units = [L.StreamUnit("weights_1", 1000, {1: [m.layers[1].weight, m.layers[1].bias]})]
        sv = _Saver(units)
        st = L.LayerStreamer(units, pause=sv.pause, resume=sv.resume, prefetch=1, device=None, pin=False)
        st.install_hooks(m.layers, 0, len(m.layers))
        lines = []
        with mock.patch.object(L.logger, "warning", lambda msg, *a: lines.append(msg % a)):
            st.stream_out(["weights_1"])
            st._out_t.clear()
            st.regain("weights_1")
        self.assertIn("out_to_regain_s=n/a", [x for x in lines if " REGAIN tag=" in x][0])
