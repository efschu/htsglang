"""Task #3 (17.09.2026), the flip tail's host->device re-admission.

xsn246 epoch 8 (D wakes at 15:43:42Z): five ~94k-token requests re-admitted
in ~4.5 s. The read side's own instrument on D-TP0:

    #1436 ARENA-GET calls=2048 pages=261965 find_ms=1037 ref_ms=101
                     resolve+pin_ms=2681

2.7 s of `resolve+pin` for an arena that is PRE-PINNED at bind (every
`pin_slots` call walks list -> unique -> index for an answer that is 0),
one `operation.increment` per PAGE, and 2048 calls of 128 pages each. The
device half had no number at all. This file pins the three fast paths and
the instrument's wiring; the device copy itself is measured on the next
boot (WEG2-LOAD-DEVICE) before it is rebuilt.
"""

from __future__ import annotations

import importlib
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402


def test_a_pre_pinned_arena_answers_pin_slots_without_a_tensor_walk():
    """`_all_pinned` short-circuits: no unique, no index, no cudart."""
    calls = {"n": 0}

    class _Pinned:
        def __getitem__(self, idx):
            calls["n"] += 1
            return torch.zeros(1, dtype=torch.bool)

    obj = types.SimpleNamespace(_pin=True, _all_pinned=True, _pinned=_Pinned())
    assert ap.ArenaMHAHostPool.pin_slots(obj, [5, 6, 7]) == 0
    assert calls["n"] == 0, "the fast path still walked the pinned mask"


def test_an_unpinned_arena_still_takes_the_slow_path():
    """Mutant guard: without the flag the walk runs (the pinned mask is read)."""
    seen = {"read": 0}

    class _Pinned:
        def __getitem__(self, idx):
            seen["read"] += 1
            return torch.ones(int(idx.numel()) if torch.is_tensor(idx) else 1, dtype=torch.bool)

        def __setitem__(self, idx, val):
            pass

    obj = types.SimpleNamespace(_pin=True, _all_pinned=False, _pinned=_Pinned(),
                                _pin_base=0, _pin_bytes=1)
    # every slot already pinned -> returns 0 after reading the mask once
    assert ap.ArenaMHAHostPool.pin_slots(obj, [5, 6, 7]) == 0
    assert seen["read"] >= 1


def test_storage_batch_is_env_tunable_and_defaults_to_the_upstream_128(monkeypatch):
    from sglang.srt.mem_cache import hicache_storage as hs
    monkeypatch.delenv("SGLANG_HICACHE_STORAGE_BATCH", raising=False)
    importlib.reload(hs)
    assert hs.STORAGE_BATCH_SIZE == 128
    monkeypatch.setenv("SGLANG_HICACHE_STORAGE_BATCH", "1024")
    importlib.reload(hs)
    assert hs.STORAGE_BATCH_SIZE == 1024
    monkeypatch.delenv("SGLANG_HICACHE_STORAGE_BATCH", raising=False)
    importlib.reload(hs)


def test_the_launcher_ships_the_larger_batch_but_an_operator_value_wins(monkeypatch):
    from sglang.srt.weg2 import launcher
    src = open(launcher.__file__).read()
    assert 'env.setdefault("SGLANG_HICACHE_STORAGE_BATCH", "1024")' in src


def test_the_load_instrument_is_wired_at_both_ends():
    """The producer stamps `_weg2_load_meta[id(finish_event)]` in
    `start_loading`; the tree's `loading_check` pops it and prints
    WEG2-LOAD-DEVICE. Source pins, because both sites run only with a live
    device pool."""
    from sglang.srt.managers import cache_controller as cc
    from sglang.srt.mem_cache import hi_mamba_radix_cache as hm
    s_cc = open(cc.__file__).read()
    s_hm = open(hm.__file__).read()
    assert "_meta[id(producer_event.finish_event)]" in s_cc and "_weg2_load_meta" in s_cc
    assert "operation.increment(len(slots) * self.page_size)" in s_cc
    assert "WEG2-LOAD-DEVICE" in s_hm
    assert "start_event.elapsed_time(finish_event)" in s_hm
