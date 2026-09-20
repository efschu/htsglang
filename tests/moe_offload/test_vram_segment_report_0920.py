"""fnFL2 v4/v5 (20.09.): the KV sizing charges the caching allocator's whole
reserve (PP1: 13.05 GiB reserved vs 5.97 allocated), so the census names the
segments whose inactive part is pinned by a live block, with the frame that
allocated the pinning block when the memory history is armed."""

import types

from sglang.srt.model_executor import vram_family_census as vc


def _snap():
    return {
        "segments": [
            {  # 1.2 GiB segment, 0.44 GiB live -> pins 0.76 GiB
                "total_size": 1200 * 2**20,
                "active_size": 440 * 2**20,
                "blocks": [
                    {"size": 440 * 2**20, "state": "active_allocated",
                     "frames": [{"filename": "/x/torch/foo.py", "line": 1, "name": "t"},
                                {"filename": "/repo/python/sglang/srt/layers/moe/expert_offload.py", "line": 5190, "name": "presplit"}]},
                    {"size": 760 * 2**20, "state": "inactive", "frames": []},
                ],
            },
            {  # fully active: not reported
                "total_size": 500 * 2**20, "active_size": 500 * 2**20,
                "blocks": [{"size": 500 * 2**20, "state": "active_allocated", "frames": []}],
            },
            {  # fully inactive: empty_cache can return it, not reported
                "total_size": 900 * 2**20, "active_size": 0,
                "blocks": [{"size": 900 * 2**20, "state": "inactive", "frames": []}],
            },
            {  # below the 64 MiB floor
                "total_size": 100 * 2**20, "active_size": 60 * 2**20,
                "blocks": [{"size": 60 * 2**20, "state": "active_allocated", "frames": []},
                           {"size": 40 * 2**20, "state": "inactive", "frames": []}],
            },
        ]
    }


def test_report_names_the_pinned_segments_and_their_holders(monkeypatch):
    monkeypatch.setattr(vc.torch.cuda.memory, "_snapshot", _snap, raising=False)
    line = vc.allocator_segment_report()
    assert line.startswith("1 segment(s) pin 0.74 GiB of inactive reserve: ")
    assert "seg 1200MiB active 440 held by 440MiB@srt/layers/moe/expert_offload.py:5190 presplit" in line


def test_report_survives_a_missing_snapshot(monkeypatch):
    def boom():
        raise RuntimeError("no cuda")

    monkeypatch.setattr(vc.torch.cuda.memory, "_snapshot", boom, raising=False)
    assert vc.allocator_segment_report().startswith("segment report unavailable")


def test_frame_str_skips_torch_internals():
    assert vc._frame_str([{"filename": "<string>", "line": 1, "name": "a"}]) == "?"
    assert vc._frame_str([{"filename": "??", "line": 0, "name": "torch::unwind::unwind()"}, {"filename": "/r/python/sglang/x.py", "line": 3, "name": "f"}]) == "x.py:3 f"
    assert vc._frame_str(None) == "?"


def test_census_emits_the_report_only_for_a_real_gap(monkeypatch, caplog):
    import logging

    calls = []
    monkeypatch.setattr(vc, "allocator_segment_report", lambda: calls.append(1) or "R")
    monkeypatch.setattr(vc, "dump_load_memsnap", lambda tag: calls.append(tag))
    alloc = {"a": 5.97 * 2**30, "r": 13.05 * 2**30}
    monkeypatch.setattr(vc.torch.cuda, "memory_allocated", lambda: alloc["a"], raising=False)
    monkeypatch.setattr(vc.torch.cuda, "memory_reserved", lambda: alloc["r"], raising=False)
    model = types.SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [])
    with caplog.at_level(logging.INFO, logger=vc.logger.name):
        vc.log_vram_family_census(model, "pp1tp0", "after load")
    assert calls == [1, "pp1tp0"]
    alloc["r"] = alloc["a"] + 0.5 * 2**30
    vc.log_vram_family_census(model, "pp1tp0", "after load")
    assert calls == [1, "pp1tp0"]  # 0.5 GiB gap: no report
