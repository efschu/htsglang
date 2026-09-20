"""SGLANG_MOE_OFFLOAD_TIMING=1: the hybrid backend times full vs linear
attention per prefill forward and logs the split when the next forward starts."""

import logging

import torch

from sglang.srt.layers.attention import hybrid_linear_attn_backend as hb


class _Ev:
    def __init__(self, t):
        self.t = t

    def elapsed_time(self, other):
        return other.t - self.t


def test_full_and_linear_halves_are_logged_per_forward(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    hb._ATTN_T.update({"on": True, "ev": {"full": [], "linear": []}, "layers": 0, "tokens": 0, "forwards": 0})
    caplog.set_level(logging.INFO, logger=hb.__name__)
    # forward 1: layers 0..3, even = linear, odd = full
    for lid in range(4):
        hb._attn_timing_begin(lid, 8192)
        hb._attn_timing_note("full" if lid % 2 else "linear", _Ev(0), _Ev(10.0 if lid % 2 else 3.0))
    assert "ATTN-TIMING-PREFILL" not in caplog.text
    hb._attn_timing_begin(0, 2432)  # next forward flushes forward 1
    lines = [r.getMessage() for r in caplog.records if "ATTN-TIMING-PREFILL" in r.getMessage()]
    assert len(lines) == 1
    assert "forward=1 tokens=8192 layers=4 full_attn_ms=20.0 (2 layers) linear_attn_ms=6.0 (2 layers)" in lines[0]
    assert hb._ATTN_T["layers"] == 1 and hb._ATTN_T["tokens"] == 2432


def test_the_switch_is_the_moe_timing_switch(monkeypatch):
    for raw, want in (("", False), ("0", False), ("1", True)):
        hb._ATTN_T["on"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_MOE_OFFLOAD_TIMING", raising=False)
        else:
            monkeypatch.setenv("SGLANG_MOE_OFFLOAD_TIMING", raw)
        assert hb._attn_timing_on() is want, raw
    hb._ATTN_T["on"] = None


def test_timing_is_off_while_a_stream_captures(monkeypatch):
    """fn7e: the verify graph runs an extend forward through forward_extend;
    inside a capture neither the flush's synchronize nor elapsed_time are
    permitted, so the timer must stand down there."""
    import inspect

    src = inspect.getsource(hb.HybridLinearAttnBackend.forward_extend)
    assert "_attn_timing_on() and not torch.cuda.is_current_stream_capturing()" in src
