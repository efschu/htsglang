# SPDX-License-Identifier: Apache-2.0
"""#1962 P-LAYER-STREAM -- the CUDA smoke review 1971 (F8) asks for before a switch-on boot.

The hermetic suite runs the streamer with "the device is the host": no side stream, no event, no
record_stream, no ``.data`` swap of a CUDA parameter under ``torch.inference_mode``, no ring peak.
This file runs exactly that half on ONE card, without D and without a server (skipped without CUDA):

* outputs with units paused are bit-identical to the resident run (the paused originals are poisoned
  with NaN, so a forward that read one would differ), over several forwards (cyclic prefetch);
* the staging ring stays bounded: the allocator's peak above the resident baseline <= staging_bytes()
  (+ one allocator segment of slack), and peak_live_sets <= prefetch + 2;
* the regain restores the content from the host image even when the "pause" destroyed it.

Run it in a gpuq window on one card (CUDA_VISIBLE_DEVICES=<card>), through pytest_gedeckelt.sh with
PYTEST_PYTHONPATH as usual -- it needs ~1.5 GiB.
"""
from __future__ import annotations

import pytest
import torch

from flliper.srt.pdflip import p_layer_stream as L

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke: needs one card")

N, W = 16, 2048


def _model(dev):
    torch.manual_seed(0)
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Linear(W, W, dtype=torch.bfloat16) for _ in range(N)]).to(dev)
    return m


def _fwd(m, x):
    for lay in m.layers:
        x = torch.tanh(lay(x))
    return x


class _PoisonSaver:
    def __init__(self, units):
        self.units = {u.tag: u for u in units}

    def pause(self, tag):
        for ts in self.units[tag].tensors.values():
            for t in ts:
                t.data.fill_(float("nan"))

    def resume(self, tag):
        pass                                              # no backup: the regain must copy the image back


def test_cuda_stream_identical_bounded_and_restored():
    L.reset_for_tests()
    dev = torch.device("cuda", 0)
    m = _model(dev)
    x = torch.randn(1024, W, dtype=torch.bfloat16, device=dev)
    with torch.inference_mode():
        ref = _fwd(m, x).clone()
    units = [L.StreamUnit("weights_1", 0, {i: [m.layers[i].weight, m.layers[i].bias] for i in range(8, 16)}),
             L.StreamUnit("weights_0", 0, {i: [m.layers[i].weight, m.layers[i].bias] for i in range(1, 8)})]
    for u in units:
        u.nbytes = sum(t.numel() * t.element_size() for ts in u.tensors.values() for t in ts)
    st = L.LayerStreamer(units, pause=_PoisonSaver(units).pause, resume=_PoisonSaver(units).resume, prefetch=2,
                         device=dev, sync=lambda: torch.cuda.synchronize(dev), empty_cache=torch.cuda.empty_cache)
    st.install_hooks(m.layers, 0, N)
    L.install(st)
    try:
        st.stream_out(["weights_1", "weights_0"])
        assert L.force_eager()
        assert torch.isnan(m.layers[3].weight).all(), "the paused original is poisoned"
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        with torch.inference_mode():
            for _ in range(4):
                out = _fwd(m, x)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) - base
        act = 3 * 1024 * W * 2                              # x, the layer output, tanh
        assert st.counters["peak_live_sets"] <= st.prefetch + 2
        assert peak <= st.staging_bytes() + act + (2 << 20), (peak, st.staging_bytes())
        assert st.counters["swapped"] == 4 * 15
        st.regain("weights_0")
        st.regain("weights_1")
        assert not L.force_eager()
        with torch.inference_mode():
            torch.testing.assert_close(_fwd(m, x), ref, rtol=0, atol=0)
    finally:
        L.reset_for_tests()
