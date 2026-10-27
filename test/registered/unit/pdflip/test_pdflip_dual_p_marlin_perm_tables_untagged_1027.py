# SPDX-License-Identifier: Apache-2.0
"""W-DUAL-P-UNTAGGED on the sm_86 P ranks (dkr27bnvfp4dualbar1drei10271237, variant 'dualm').

PP1/PP2 (the 3080s) run ``marlin_native_inplace``; PP0 (5090, flashinfer_cutlass) does not,
and the 11:55 W4A8 boot (``w4a8_int8``) passed the same riegel. The first
``nvfp4_marlin_inplace._scale_perms('cuda:N')`` call happens during the load of the dual shared
part, inside ``transient_load_scope``, i.e. in the load pool. The ``lru_cache`` keeps the two
tables (512 B each) for the life of the process, the after-load release keeps the pool
(``live=0.0 MiB``: < 0.05 MiB) and ``assert_no_untagged_live`` stops the boot with
``0.0 MiB`` (tolerance default 0, ``live > tolerance`` is true for any residue).

Pinned here (CPU, hermetic):
* the cached permutation tables are born under ``back_into_tag_pool(force=True)`` (so inside a
  transient scope they land under the weights tag and sleep/wake with P);
* outside any stepped-out block nothing changes (back_into_tag_pool is a no-op there);
* the riegel line names the residue in KiB and the KEPT line names bytes and block count, so
  a tiny residue no longer reads as 0.0 MiB.
"""

from __future__ import annotations

import contextlib
import logging
import types

import pytest
import torch

from flliper.srt.layers.quantization import nvfp4_marlin_inplace as M
from flliper.srt.managers import pdflip_memory_saver as S
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Recorder:
    """back_into_tag_pool stand-in plus a ``Tensor.to`` spy: which device-moving calls ran
    inside a (forced) back-into-pool block."""

    def __init__(self, monkeypatch):
        self.depth = 0
        self.forced = []
        self.to_calls = []  # (inside_pool, device)
        rec = self

        @contextlib.contextmanager
        def back_into_tag_pool(force=False):
            rec.forced.append(force)
            rec.depth += 1
            try:
                yield True
            finally:
                rec.depth -= 1

        monkeypatch.setattr(S, "back_into_tag_pool", back_into_tag_pool)
        orig_to = torch.Tensor.to

        def spy_to(self_t, *a, **kw):
            rec.to_calls.append((rec.depth > 0, a[0] if a else kw.get("device")))
            return orig_to(self_t, *a, **kw)

        monkeypatch.setattr(torch.Tensor, "to", spy_to)


@pytest.fixture(autouse=True)
def _fresh_caches():
    M._scale_perms.cache_clear()
    M._weight_perms.cache_clear()
    yield
    M._scale_perms.cache_clear()
    M._weight_perms.cache_clear()


def test_the_scale_permutation_tables_are_born_under_the_weights_tag(monkeypatch):
    rec = _Recorder(monkeypatch)
    perm, inv = M._scale_perms("cpu")
    assert perm.numel() == 64 and inv.numel() == 64
    assert rec.to_calls, "the tables must be moved to the device"
    assert all(inside for inside, _ in rec.to_calls), rec.to_calls
    assert rec.forced and all(rec.forced), "a transient scope keeps nested non-forced entries OUT of the pool"
    n = len(rec.to_calls)
    M._scale_perms("cpu")  # cached: no second allocation
    assert len(rec.to_calls) == n


def test_the_weight_permutation_tables_are_born_under_the_weights_tag(monkeypatch):
    rec = _Recorder(monkeypatch)
    perm, inv = M._weight_perms("cpu")
    assert perm.numel() == 1024 and inv.numel() == 1024
    assert rec.to_calls and all(inside for inside, _ in rec.to_calls), rec.to_calls
    assert all(rec.forced)


def test_the_tables_stay_correct_values(monkeypatch):
    _Recorder(monkeypatch)
    perm, inv = M._weight_perms("cpu")
    assert torch.equal(inv[perm], torch.arange(perm.numel()))
    sp, sinv = M._scale_perms("cpu")
    assert torch.equal(sinv[sp], torch.arange(sp.numel()))


def test_without_a_stepped_out_block_the_real_back_into_tag_pool_is_a_noop(monkeypatch):
    monkeypatch.setattr(S, "_STEPPED_OUT_POOLS", [])
    with S.back_into_tag_pool(force=True) as back:
        assert back is False
    perm, inv = M._scale_perms("cpu")  # real context manager, no pool: plain allocation
    assert perm.device.type == "cpu" and inv.numel() == 64


def test_the_riegel_line_names_a_sub_mib_residue_in_kib():
    # dkr27bnvfp4dualbar1drei10271237: 'W-DUAL-P-UNTAGGED: 0.0 MiB ... (load=0.0 MiB)'
    with pytest.raises(S.PdFlipDualPUntagged) as ei:
        S.assert_no_untagged_live({"load": 1.0 / 1024.0})
    msg = str(ei.value)
    assert "0.0 MiB (1.0 KiB)" in msg, msg
    S.assert_no_untagged_live({"load": 0.0})


def test_the_kept_line_names_bytes_and_blocks(monkeypatch, caplog):
    snap = [{"blocks": [{"size": 512, "state": "active_allocated"},
                        {"size": 512, "state": "active_allocated"},
                        {"size": 2 << 20, "state": "inactive"}]}]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda pool_id=None: snap)

    def boom(*a, **k):
        raise RuntimeError("cannot tell")

    monkeypatch.setattr(torch._C, "_cuda_checkPoolLiveAllocations", boom, raising=False)
    with caplog.at_level(logging.WARNING, logger=S.logger.name):
        assert S._pool_has_live_blocks(types.SimpleNamespace(id=3), "after-load pool=load") is True
    line = [r.getMessage() for r in caplog.records if "transient pool KEPT" in r.getMessage()][-1]
    assert "live=0.0 MiB (1024 B in 2 block(s))" in line, line
