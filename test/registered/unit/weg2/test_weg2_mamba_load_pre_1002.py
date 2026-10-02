"""DP-NACHLAUF 02.10.: the mamba component of WEG2-START-LOADING names its
untimed part. N5d (0c996cf05c 1002_124821, D->P epoch 25): components
mamba=345 ms against the timed sub-stages idx 0 + issue 2 + select 0 + split 7
-- 336 ms nobody named, on the scheduler thread right before P's first prefill
forward. The arena/staging split of the host rows runs on every layer call
before the layer>0 early return; it is now summed as mamba.pre (+ pre_n
calls). Instrument only (red before: no `pre` stamp)."""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp  # noqa: E402


def test_the_split_before_the_early_return_is_timed():
    src = inspect.getsource(amp.ArenaMambaPoolHost.load_to_device_per_layer)
    i_pre = src.index('_subp["pre"]')
    i_key = src.index("if layer_id != 0 and self._state_loaded_key == key:")
    assert src.index("_tpre = time.perf_counter()") < src.index("hi, is_arena = self._split(host_indices)") < i_pre < i_key
    assert '_subp["pre_n"]' in src


def test_the_staging_rows_per_layer_copies_are_timed():
    src = inspect.getsource(amp.ArenaMambaPoolHost.load_to_device_per_layer)
    assert src.count('_subp["rest"]') == 2 and src.count("_trest = time.perf_counter()") == 2
