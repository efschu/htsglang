# SPDX-License-Identifier: Apache-2.0
"""L3P clear guard (review 28.09.): a clear of the persistent L3 store is refused
without force (W166, named), the planner's cold-prefill flush no longer asks for
it, and the force reaches HiCacheFile.clear through every tree's
clear_storage_backend. Hermetic: HiCacheFile.clear on an object double."""

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache import hicache_storage as hs  # noqa: E402


def _store(tmp_path, persistent=True):
    d = tmp_path / "l3-qwen27b-Qwen3.8-27B-INT8-abc"
    (d / "ab").mkdir(parents=True)
    (d / "ab" / "abcd.bin").write_bytes(b"x" * 64)
    if persistent:
        (d / "L3_IDENTITY.json").write_text("{}")
    f = object.__new__(hs.HiCacheFile)
    f.file_path = str(d)
    f._l3idx, f._l3idx_tried = None, True
    f._evictor = types.SimpleNamespace(clear=lambda: None)
    f.metadata_cache = None
    return f, d


def test_a_persistent_store_refuses_a_clear_without_force(tmp_path, caplog):
    f, d = _store(tmp_path)
    with caplog.at_level(logging.ERROR):
        assert f.clear() is False
    assert (d / "ab" / "abcd.bin").exists(), "not a page removed"
    assert "W166 PdFlipL3ClearRefused" in caplog.text


def test_force_clears_it(tmp_path):
    f, d = _store(tmp_path)
    assert f.clear(force=True) is True
    assert not (d / "ab" / "abcd.bin").exists()


def test_a_per_boot_store_clears_as_before(tmp_path):
    f, d = _store(tmp_path, persistent=False)
    assert f.clear() is True
    assert not (d / "ab" / "abcd.bin").exists()


def test_clear_storage_passes_force_and_reads_the_refusal():
    calls = []

    class B:
        def clear(self, force=False):
            calls.append(force)
            return not (not force)

    class Upstream:  # a backend whose clear takes no argument and returns None
        def clear(self):
            calls.append("up")

    assert hs.clear_storage(B()) is False
    assert hs.clear_storage(B(), force=True) is True
    assert hs.clear_storage(Upstream(), force=True) is True
    assert calls == [False, True, "up"]


def test_the_request_carries_force_and_the_planner_flushes_the_tree_only():
    from flliper.srt.managers.io_struct import ClearHiCacheReqInput
    from flliper.srt.planner.energy import CACHE_FLUSH_ENDPOINTS

    assert ClearHiCacheReqInput().force is False
    assert ClearHiCacheReqInput(force=True).force is True
    assert CACHE_FLUSH_ENDPOINTS == ("/flush_cache",)


def test_the_scheduler_reports_the_refusal():
    from flliper.srt.managers.io_struct import ClearHiCacheReqInput
    from flliper.srt.managers.scheduler import Scheduler

    seen = []
    tree = types.SimpleNamespace(clear_storage_backend=lambda force=False: seen.append(force) or force)
    h = types.SimpleNamespace(enable_hierarchical_cache=True, tree_cache=tree)
    assert Scheduler.clear_hicache_storage_wrapped(h, ClearHiCacheReqInput()).success is False
    assert Scheduler.clear_hicache_storage_wrapped(h, ClearHiCacheReqInput(force=True)).success is True
    assert seen == [False, True]
