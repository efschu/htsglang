"""#1402 (boot xsn132, 2026-09-15): a completing canonical extent write no
longer fsyncs every 32 KiB page by default; SGLANG_HICACHE_CANONICAL_FSYNC=1
turns the per-page fsync back on. The rename-on-complete protocol is untouched.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.mem_cache import canonical_page_store as cps


def _window(total=64):
    return cps.CanonicalExtentWindow(
        label="test", total_bytes=total, extents=((0, total),)
    )


def _write(tmp_path, monkeypatch, env, explicit=None):
    if env is None:
        monkeypatch.delenv("SGLANG_HICACHE_CANONICAL_FSYNC", raising=False)
    else:
        monkeypatch.setenv("SGLANG_HICACHE_CANONICAL_FSYNC", env)
    calls = []
    monkeypatch.setattr(cps.os, "fsync", lambda fd: calls.append(fd))
    final = os.path.join(str(tmp_path), "ab", "abcd.bin")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    payload = torch.arange(64, dtype=torch.uint8)
    kw = {} if explicit is None else {"fsync": explicit}
    res = cps.write_extents(final, _window(), payload, **kw)
    assert res.completed and os.path.exists(final)
    assert open(final, "rb").read() == bytes(range(64)), "bytes land either way"
    return len(calls)


def test_default_is_no_fsync_per_page(tmp_path, monkeypatch):
    assert _write(tmp_path, monkeypatch, None) == 0
    assert _write(tmp_path / "b", monkeypatch, "0") == 0


def test_env_switches_the_fsync_back_on(tmp_path, monkeypatch):
    assert _write(tmp_path, monkeypatch, "1") == 1


def test_explicit_argument_wins_over_the_env(tmp_path, monkeypatch):
    assert _write(tmp_path, monkeypatch, "1", explicit=False) == 0
    assert _write(tmp_path / "b", monkeypatch, "0", explicit=True) == 1
