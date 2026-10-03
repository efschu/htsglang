"""MAMBA-SNAPSHOT-FENCE, NF port (01.10.): the H2 device-index form is fenced too.

On the NF line the mamba anchor's host pool is the arena (ArenaMambaHostPool,
MAMBA-ARENA): ``backup_accepts_device_indices`` answers "" for an all-arena
op, so the live write-back takes the H2 ON-CARD branch of
``HybridCacheController.start_writing`` (``backup_from_device_indices``: the
xsn351 pointer kernel on the write stream, plus the H63c PLE side states on
the same stream) -- not the ``backup_from_device_all_layer`` branch the 27B
test drives. The fence sits after both branches; this pins the branch NF
actually runs. RED on 81c5826004 (no fence on either branch)."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc  # noqa: E402

from test_mamba_snapshot_fence_n2_1001 import _controller, _Dev, _Log, _mamba  # noqa: E402


def _run_on_card(monkeypatch, pool_transfers):
    log = _Log()
    monkeypatch.setattr(hcc, "device_module", _Dev(log))
    monkeypatch.setattr(hcc, "consume_gate", lambda *a, **kw: True)
    c = _controller(log, pool_transfers)
    c._device_index_write_refusal = lambda op: ""   # H2: every pool takes card indices
    c.start_writing()
    log.append(("forward", "compute", "next"))
    return log, c


def test_the_on_card_arena_write_is_fenced(monkeypatch):
    """The MAMBA-ARENA path (all-arena op, H2 on-card form): the next forward
    waits for the op's finish event, which is recorded after the copy."""
    log, c = _run_on_card(monkeypatch, _mamba())
    i_copy = log.index(("copy", "write", "indices"))
    i_fin = log.index(("record", "finish", "current"))
    i_fwd = log.index(("forward", "compute", "next"))
    assert i_copy < i_fin < i_fwd
    waits = [k for k, e in enumerate(log) if e == ("wait", "compute", "finish")]
    assert waits and i_fin < waits[0] < i_fwd, log
    assert getattr(c, "_mamba_fence_n", 0) == 1


def test_an_on_card_kv_only_write_stays_unordered(monkeypatch):
    log, _ = _run_on_card(monkeypatch, None)
    assert ("copy", "write", "indices") in log
    assert ("wait", "compute", "finish") not in log


def test_a_mamba_read_placeholder_is_not_a_snapshot():
    """A MAMBA transfer without device rows (PREFETCH / BACKUP_STORAGE shape:
    host rows + keys only) reads no device state and needs no fence."""
    assert not hcc._mamba_snapshot_fence_needed(
        [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]),
                      keys=["k"])])
