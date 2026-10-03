# SPDX-License-Identifier: Apache-2.0
"""+254 MiB in P's sleep (NF P snapshots 09280831, PP0 sleep6/sleep7): the
staging buffers the serving phase keeps cached are dropped at the sleep and
re-created lazily (a), the static-state stash sleeps on the host (b), and the
holder report names who references the largest untagged blocks (c)."""

import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import sleep_staging as ss  # noqa: E402


class _Pool:
    """weakref-able like ArenaMambaPoolHost (the registry is a WeakSet)."""

    def __init__(self, stage):
        self._state_dev_stage = stage


def test_free_staging_drops_the_three_lazy_stages(monkeypatch):
    from sglang.srt.layers.moe import expert_offload as eo

    pool = _Pool(torch.empty((4, 1024), dtype=torch.uint8))
    ss.register_mamba_pool(pool)
    ple = torch.nn.Module()
    ple._eager_prefetch_buffer = torch.empty((8, 256), dtype=torch.uint8)
    model = torch.nn.Sequential(ple)
    monkeypatch.setattr(eo, "_GATHER_RINGS", {("cpu", torch.uint8, (16,)): [torch.empty((32, 16), dtype=torch.uint8), None]})
    freed = ss.free_staging(model, env={})
    assert freed == {"mamba_state_stage": 4096, "ple_prefetch_buffer": 2048, "expert_gather_ring": 512}
    assert pool._state_dev_stage is None and ple._eager_prefetch_buffer is None and not eo._GATHER_RINGS
    assert "6.5 KiB" not in ss.format_freed(freed) and ss.format_freed(freed).startswith("0.0 MiB (")


def test_switch_off_keeps_everything(monkeypatch):
    pool = _Pool(torch.empty(8, dtype=torch.uint8))
    ss.register_mamba_pool(pool)
    assert ss.free_staging(None, env={ss.ENV: "0"}) == {}
    assert pool._state_dev_stage is not None


def test_a_dropped_mamba_stage_is_recreated_on_first_use():
    # the load path's own guard: None -> allocate (arena_mamba_pool
    # _load_states_all_layers); pinned by reading the shipped condition
    import inspect

    from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp

    src = inspect.getsource(amp)
    assert 'getattr(self, "_state_dev_stage", None) is None' in src
    assert "_ss.register_mamba_pool(self)" in src


def test_static_state_stash_sleeps_on_the_host_and_restores():
    from sglang.srt.managers.scheduler_components.weight_updater import _import_static_state

    m = torch.nn.Module()
    m.register_buffer("cos", torch.arange(6, dtype=torch.float32))
    stash = ss.export_static_state_host(m)
    assert all(t.device.type == "cpu" for _n, t in stash["buffers"])
    m.cos.zero_()
    _import_static_state(m, stash)
    assert torch.equal(m.cos, torch.arange(6, dtype=torch.float32))


def test_holder_description_names_class_and_attribute():
    class Ring:
        pass

    obj = object()
    r = Ring()
    r.last_mbs = [None, obj]
    got = ss._describe(obj, 3, set())
    assert any("Ring.last_mbs" in g for g in got), got


def test_holder_report_is_bounded_and_empty_without_a_snapshot():
    assert ss.holder_report(None) == []
    snap = {"segments": [{"address": 0, "blocks": [
        {"size": 1 << 20, "state": "active_allocated",
         "frames": [{"filename": "/x/sglang/srt/kernels/hc_combine.py", "line": 86, "name": "hc_combine"}]},
        {"size": 1 << 30, "state": "active_allocated", "frames": []},
    ]}]}
    lines = ss.holder_report(snap, top=4)
    assert len(lines) == 1 and "srt/kernels/hc_combine.py:86 hc_combine" in lines[0]


def test_the_sleep_hooks_are_wired():
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    assert "self._weg2_free_sleep_staging()\n        self._weg2_log_sleep_residue(census, tags)" in src
    assert "_ss.export_static_state_host(self.tp_worker.model_runner.model)" in src
    assert "WEG2-SLEEP-HOLDER" in src
