"""FLIP-LEGS 02.10.: the serial work before the sleeper's first deposit.

MEASURED (N4p f405217a61 / N4q 58361d5471): between the front issuing the
gathered legs and the first WEG2-SLEEP-TAG-TIME t0, every sleeping rank spends
58-72 ms p50 in both directions (the waking P ranks are ready after 5-6 ms).
WEG2-SLEEP-PRELOOP names it: census_credit 41-54 ms p50 on every sleeping
rank, the rest a few ms. census_credit holds the static-state stash export
(``export_static_state_host``: one pageable, synchronizing ``.to("cpu")`` per
model buffer), the TP barrier (waits for the slowest rank's export) and the
credit/census reads. The matching import in the waker's tail measures
static_import 16-79 ms (WEG2-WAKE-TAIL-SUB).

Pinned here (red before): the stash copies into pinned host memory with
non_blocking=True and synchronizes ONCE per device (stage_buffers), the values
and names are those of the old form, the switch restores it, and the leg
splits census_credit into static_export / tp_barrier on the PRELOOP line and
prints WEG2-STATIC-EXPORT per sleep.
"""
from __future__ import annotations

import inspect
import os

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import sleep_staging as ss  # noqa: E402


def test_switch_default_on():
    assert ss.stash_pinned_on({})
    for off in ("0", "false", "no", "off"):
        assert not ss.stash_pinned_on({ss.ENV_STASH_PINNED: off})


def test_stage_buffers_syncs_once_per_device_after_the_last_copy():
    events = []

    def to_host(b):
        events.append(("copy", b))
        return f"h{b}", ("dev0" if b < 3 else "dev1")

    def sync(dev):
        events.append(("sync", dev))

    out = ss.stage_buffers([(f"n{i}", i) for i in range(5)], to_host=to_host, sync_device=sync)
    assert out == [(f"n{i}", f"h{i}") for i in range(5)]
    assert [e for e in events if e[0] == "sync"] == [("sync", "dev0"), ("sync", "dev1")]
    assert events.index(("sync", "dev0")) > events.index(("copy", 4)), events


def test_a_host_buffer_needs_no_sync():
    syncs = []
    out = ss.stage_buffers([("a", 1)], to_host=lambda b: (b, None), sync_device=syncs.append)
    assert out == [("a", 1)] and syncs == []


def test_export_values_names_and_line_numbers(monkeypatch):
    from sglang.srt.managers.scheduler_components.weight_updater import _import_static_state

    m = torch.nn.Module()
    m.register_buffer("cos", torch.arange(6, dtype=torch.float32))
    m.register_buffer("mask", torch.ones(2, 3, dtype=torch.bool).t())   # non-contiguous
    for env in ("1", "0"):
        monkeypatch.setenv(ss.ENV_STASH_PINNED, env)
        stash = ss.export_static_state_host(m)
        assert [n for n, _ in stash["buffers"]] == ["cos", "mask"]
        assert all(t.device.type == "cpu" for _n, t in stash["buffers"])
        assert ss.LAST_EXPORT["n"] == 2
        assert ss.LAST_EXPORT["bytes"] == 6 * 4 + 6
        assert ss.LAST_EXPORT["mode"] == ("pinned-async" if env == "1" else "pageable")
        m.cos.zero_()
        _import_static_state(m, stash)
        assert torch.equal(m.cos, torch.arange(6, dtype=torch.float32))


def test_the_leg_names_the_split_and_the_export():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    i_exp = src.index("_ss.export_static_state_host(self.tp_worker.model_runner.model)")
    i_line = src.index('"WEG2-STATIC-EXPORT n=%d bytes=%d ms=%.1f mode=%s"')
    i_ph1 = src.index('_weg2_ph("static_export")')
    i_bar = src.index("torch.distributed.barrier(self.tp_cpu_group)", i_ph1)
    i_ph2 = src.index('_weg2_ph("tp_barrier")')
    i_cc = src.index('_weg2_ph("census_credit")')
    assert i_exp < i_line < i_ph1 < i_bar < i_ph2 < i_cc
