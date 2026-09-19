"""SGLANG_MOE_OFFLOAD_TIMING=1 on the expert-major prefill path: per-wave
(fetch, apply) CUDA-event triples are collected per layer and the forward's
split is logged once the next forward starts (layer 0 again)."""

import logging

import torch

from sglang.srt.layers.moe import expert_offload as eo


class _Ev:
    def __init__(self, t):
        self.t = t

    def elapsed_time(self, other):
        return other.t - self.t


def _triple(t0, fetch_ms, apply_ms):
    return (_Ev(t0), _Ev(t0 + fetch_ms), _Ev(t0 + fetch_ms + apply_ms))


def test_the_prefill_split_is_logged_per_forward(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    eo._WAVE_TP.update({"ev": [], "layers": 0, "waves": 0, "spill": 0, "tokens": 0, "forwards": 0})
    caplog.set_level(logging.INFO, logger=eo.__name__)
    # forward 1: three layers, two waves each
    for lid in (0, 1, 2):
        eo._wave_timing_note_prefill(lid, [_triple(0, 10.0, 5.0), _triple(20, 30.0, 7.0)], 12, 8192)
    assert "MOE-OFFLOAD-TIMING-PREFILL" not in caplog.text  # nothing flushed yet
    # forward 2 starts at layer 0 -> forward 1 is flushed
    eo._wave_timing_note_prefill(0, [_triple(0, 1.0, 1.0)], 3, 2432)
    lines = [r.getMessage() for r in caplog.records if "TIMING-PREFILL" in r.getMessage()]
    assert len(lines) == 1
    assert "forward=1 tokens=8192 layers=3 waves=6 spill_experts=36 fetch_ms=120.0 apply_ms=36.0" in lines[0]
    # the accumulator now holds only forward 2
    assert eo._WAVE_TP["layers"] == 1 and eo._WAVE_TP["waves"] == 1 and eo._WAVE_TP["tokens"] == 2432


def test_the_expert_major_path_records_events_only_when_timing_is_on():
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(eo.MoEExpertOffloadCache._run_waves_expert_major))
    assert "_wave_timing_note_prefill(" in src and "_wave_timing_on()" in src
    assert src.count("torch.cuda.Event(enable_timing=True)") == 3
    ast.parse(src)


def test_wave_token_slice_is_env_gated_and_bounded(monkeypatch):
    """SGLANG_MOE_OFFLOAD_WAVE_SLICE bounds the grouped GEMM of one expert-major
    wave to that many (token, expert) pairs; 0 = whole wave; default 81920 =
    8192 tokens x top-k 10 (the shape that fits a 3080's Marlin workspace)."""
    import ast
    import inspect
    import textwrap

    from sglang.srt.layers.moe import expert_offload as eo

    for raw, want in (("", 81920), ("0", 0), ("4096", 4096), ("-5", 0), ("abc", 81920)):
        eo._WAVE_SLICE["pairs"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_MOE_OFFLOAD_WAVE_SLICE", raising=False)
        else:
            monkeypatch.setenv("SGLANG_MOE_OFFLOAD_WAVE_SLICE", raw)
        assert eo.wave_token_slice_pairs() == want, raw
    eo._WAVE_SLICE["pairs"] = None
    src = textwrap.dedent(inspect.getsource(eo.MoEExpertOffloadCache._run_waves_expert_major))
    ast.parse(src)
    assert "wave_token_slice_pairs()" in src
    assert "for start in range(0, int(idx_np.size), step):" in src
    # the wave is fetched ONCE, outside the slice loop
    assert src.index("self._fetch(fetch_plan)") < src.index("for start in range(0, int(idx_np.size), step):")


def test_partials_mode_is_env_gated_and_defaults_to_the_table(monkeypatch):
    """SGLANG_MOE_OFFLOAD_PARTIALS=stream drops the [T*K, H] partials table
    (1.7 GB at a 32k chunk, fn6v OOM) for an fp32 [T, H] accumulator; the
    default keeps the byte-identical combine."""
    import ast
    import inspect
    import textwrap

    from sglang.srt.layers.moe import expert_offload as eo

    for raw, want in (("", "table"), ("table", "table"), ("stream", "stream"), ("STREAM", "stream"), ("x", "table")):
        eo._PARTIALS_MODE["mode"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_MOE_OFFLOAD_PARTIALS", raising=False)
        else:
            monkeypatch.setenv("SGLANG_MOE_OFFLOAD_PARTIALS", raw)
        assert eo.partials_mode() == want, raw
    eo._PARTIALS_MODE["mode"] = None
    src = textwrap.dedent(inspect.getsource(eo.MoEExpertOffloadCache._run_waves_expert_major))
    ast.parse(src)
    assert "out_acc.index_add_(0, rows, part.to(torch.float32))" in src
    assert "combine_topk_partials(partials.view(T, K, -1), out_full, saved_rsf)" in src
