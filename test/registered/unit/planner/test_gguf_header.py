# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Header-only GGUF resolution for HF-hub repos.

Sizing a hub GGUF must fetch ONLY the header (a ranged read of the first tens
of MB), never the multi-GB tensor data -- a full download hangs the offline
planner (the live "PLAN REJECTED ... pick one" symptom was the stale verdict
sitting on screen while a 20 GB download ran).
"""
import os
import struct

import huggingface_hub

from sglang.srt.planner import model as M

# Minimal valid GGUF header: magic + version=3 + n_tensors=0 + n_kv=0.
MINIMAL_GGUF = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)


class _FakeFile:
    def __init__(self, data):
        self._d = data

    def read(self, n):
        return self._d[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeFS:
    def __init__(self, data, calls):
        self._d = data
        self._calls = calls

    def open(self, remote, mode="rb"):
        self._calls.append(remote)
        return _FakeFile(self._d)


def test_download_gguf_header_fetches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_gguf_header_cache_dir", lambda: str(tmp_path))
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "HfFileSystem", lambda *a, **k: _FakeFS(MINIMAL_GGUF, calls)
    )

    p = M._download_gguf_header("org/repo-GGUF", "model-Q4_K_M.gguf")
    assert os.path.isfile(p)
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"
    assert calls == ["org/repo-GGUF/model-Q4_K_M.gguf"]  # fetched exactly once

    # A second resolve of the same quant reuses the cached header, no re-fetch.
    p2 = M._download_gguf_header("org/repo-GGUF", "model-Q4_K_M.gguf")
    assert p2 == p
    assert len(calls) == 1


def test_download_gguf_header_basename_only(tmp_path, monkeypatch):
    # A caller may pass "repo/file.gguf"; only the basename is the hub filename.
    monkeypatch.setattr(M, "_gguf_header_cache_dir", lambda: str(tmp_path))
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "HfFileSystem", lambda *a, **k: _FakeFS(MINIMAL_GGUF, calls)
    )
    M._download_gguf_header("org/repo-GGUF", "sub/dir/model-Q6_K.gguf")
    assert calls == ["org/repo-GGUF/model-Q6_K.gguf"]


def test_resolve_gguf_choice_routes_through_header(monkeypatch):
    # resolve_model_ref must size a hub GGUF via the header path, never a full
    # weight download.
    seen = {}

    def _hdr(repo, fn):
        seen["hdr"] = (repo, fn)
        return "/tmp/fake.header"

    monkeypatch.setattr(M, "_download_gguf_header", _hdr)

    def _boom(*a, **k):
        raise AssertionError("full hf_hub_download must not run for a GGUF choice")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)

    out = M.resolve_model_ref("org/repo-GGUF", gguf_choice="model-Q4_K_M.gguf")
    assert out == "/tmp/fake.header"
    assert seen["hdr"] == ("org/repo-GGUF", "model-Q4_K_M.gguf")
