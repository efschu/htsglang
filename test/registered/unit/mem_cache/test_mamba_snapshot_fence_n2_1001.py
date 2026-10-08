"""MAMBA-SNAPSHOT-FENCE (N2, 01.10.): no forward may overtake a recurrent-state snapshot.

27B needle-MISS: the persisted needle anchor (fe9e35.mamba, boot 10010740) is
a statistically PLAUSIBLE GDN state (l3_entry_diff stats: every layer and TP
share in the range of healthy blobs, no zero / NaN share) that resumes to a
wrong answer, while the same prompt's entry written at 00:57 resumed right --
"a valid state of another moment", timing-dependent.

The write-back (HybridCacheController.start_writing) orders the D2H copy
AFTER the compute stream (``start_event``: the write stream waits for it) but
never the compute stream after the copy: the ack only tells the HOST. KV rows
are append-only, so a later forward cannot change what a KV copy reads. The
recurrent state is updated IN PLACE: a forward launched after the issue can
rewrite the state rows while the async D2H still reads them, and the arena /
L3 then hold the state of a later position (or a verify step later discarded)
under this node's key -- the exact symptom.

The fence: a write op that carries a MAMBA transfer makes the compute stream
wait for the op's finish event. The test drives the REAL start_writing with a
recording device module (CPU, no CUDA): the order of stream operations is the
evidence."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc  # noqa: E402


class _Log(list):
    pass


class _Event:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def record(self, stream=None):
        self.log.append(("record", self.name, getattr(stream, "name", "current")))

    def wait(self, stream):
        self.log.append(("wait", stream.name, self.name))


class _Stream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_event(self, ev):
        self.log.append(("wait", self.name, ev.name))

    def query(self):  # hicache_write_path.compute_stream_busy
        return True


class _Ctx:
    def __init__(self, log, stream):
        self.log, self.stream = log, stream

    def __enter__(self):
        self.log.append(("enter", self.stream.name))

    def __exit__(self, *a):
        self.log.append(("exit", self.stream.name))


class _Dev:
    def __init__(self, log):
        self.log = log
        self.compute = _Stream(log, "compute")
        self._n = 0

    def Event(self):  # noqa: N802
        self._n += 1
        return _Event(self.log, ("start", "finish")[(self._n - 1) % 2])

    def stream(self, s):
        return _Ctx(self.log, s)

    def current_stream(self):
        return self.compute


class _HostPool:
    def __init__(self, log):
        self.log = log

    def backup_from_device_all_layer(self, *a, **kw):
        self.log.append(("copy", "write", "all_layer"))

    def backup_from_device_indices(self, *a, **kw):
        self.log.append(("copy", "write", "indices"))


def _controller(log, pool_transfers):
    c = object.__new__(hcc.HybridCacheController)
    op = hcc.CacheOperation(torch.arange(4), torch.arange(4), 7, None,
                            pool_transfers=pool_transfers)
    c.write_queue = [op]
    c.ack_write_queue = []
    c.io_backend = "direct"
    c.mem_pool_host = _HostPool(log)
    c.mem_pool_device = object()
    c.write_stream = _Stream(log, "write")
    c._device_index_write_refusal = lambda op: "test"
    c.move_hybrid_indices = lambda op: (op.host_indices, op.device_indices, op.pool_transfers)
    c._dcp_kv_transfer_pairs = lambda h, d: (h, d)
    c.draft_tier_armed = lambda kind: False
    c._record_transfer_indices_on_stream = lambda *a, **kw: None
    return c


def _run(monkeypatch, pool_transfers):
    log = _Log()
    monkeypatch.setattr(hcc, "device_module", _Dev(log))
    monkeypatch.setattr(hcc, "consume_gate", lambda *a, **kw: True)
    c = _controller(log, pool_transfers)
    c.start_writing()
    log.append(("forward", "compute", "next"))   # whatever the scheduler launches next
    return log, c


def _mamba():
    return [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]),
                         device_indices=torch.tensor([5]))]


def test_the_next_forward_waits_for_a_mamba_snapshot(monkeypatch):
    """RED on dae863c7c8: the compute stream never waits for the copy -- the
    'forward' is free to run while the write stream still reads the state."""
    log, c = _run(monkeypatch, _mamba())
    i_copy = log.index(("copy", "write", "all_layer"))
    i_fin = log.index(("record", "finish", "current"))
    i_fwd = log.index(("forward", "compute", "next"))
    assert i_copy < i_fin < i_fwd
    waits = [k for k, e in enumerate(log) if e == ("wait", "compute", "finish")]
    assert waits and i_fin < waits[0] < i_fwd, log
    assert getattr(c, "_mamba_fence_n", 0) == 1


def test_a_kv_only_write_stays_unordered(monkeypatch):
    """KV rows are append-only: a KV-only write keeps the old overlap."""
    log, _ = _run(monkeypatch, None)
    assert ("wait", "compute", "finish") not in log


def test_the_switch_turns_the_fence_off(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SNAPSHOT_FENCE", "0")
    log, _ = _run(monkeypatch, _mamba())
    assert ("wait", "compute", "finish") not in log


def test_an_empty_mamba_transfer_needs_no_fence():
    assert not hcc._mamba_snapshot_fence_needed(
        [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([], dtype=torch.int64),
                      device_indices=torch.tensor([], dtype=torch.int64))])
    assert hcc._mamba_snapshot_fence_needed(_mamba())


# --------------------------------------------------------------------------- STAGE
# MAMBA-SNAPSHOT-STAGE (N2 01.10.): the fence cost P -26 % prefill (N2 boot
# 10011623 PP0: a 512-token chunk 190 ms instead of 90, bubble up to 190 ms) --
# the compute stream waited for the whole async D2H of every chunk anchor. Now
# the state rows are copied D2D into a staging buffer on the compute stream
# BEFORE start_event (the snapshot point), the D2H reads the staging copy, and
# the compute stream never waits for the D2H.


class _StagingGroup(_HostPool):
    def __init__(self, log):
        super().__init__(log)
        self.released = []

    def snapshot_mamba_transfers(self, pool_transfers):
        self.log.append(("snapshot", "compute", "d2d"))
        staged = [PoolTransfer(name=t.name, host_indices=t.host_indices,
                               device_indices=torch.arange(int(t.device_indices.numel())))
                  for t in pool_transfers]
        return staged, [(self, 0)]

    def snapshot_release(self, tok, ev):
        self.released.append((tok, ev.name))


def test_staged_snapshot_precedes_start_and_the_forward_never_waits_for_the_d2h(monkeypatch):
    """RED on 3db208a976 (and da33b3c551): the compute stream waits for
    `finish` -- the D2H -- on every MAMBA write op."""
    log = _Log()
    monkeypatch.setattr(hcc, "device_module", _Dev(log))
    monkeypatch.setattr(hcc, "consume_gate", lambda *a, **kw: True)
    c = _controller(log, _mamba())
    c.mem_pool_host = _StagingGroup(log)
    c.start_writing()
    log.append(("forward", "compute", "next"))
    i_snap = log.index(("snapshot", "compute", "d2d"))
    i_start = log.index(("record", "start", "current"))
    i_copy = log.index(("copy", "write", "all_layer"))
    assert i_snap < i_start < i_copy                      # the D2H reads the snapshot
    assert ("wait", "compute", "finish") not in log      # no wait for the D2H
    assert c.mem_pool_host.released == [(0, "finish")]   # the slot is reusable after the D2H


def test_an_unstageable_op_keeps_the_fence(monkeypatch):
    log = _Log()
    monkeypatch.setattr(hcc, "device_module", _Dev(log))
    monkeypatch.setattr(hcc, "consume_gate", lambda *a, **kw: True)
    c = _controller(log, _mamba())
    grp = _StagingGroup(log)
    grp.snapshot_mamba_transfers = lambda pt: None
    c.mem_pool_host = grp
    c.start_writing()
    assert ("wait", "compute", "finish") in log


def _arena_mamba_host(pending_rows=(0,), staging_rows=3):
    from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost

    hp = object.__new__(ArenaMambaPoolHost)
    hp.arena = object()
    hp.staging_rows = staging_rows
    hp.arena_slots = 8
    hp._pending = {r: (1, True) for r in pending_rows}
    return hp


class _Dev2:
    def __init__(self, L=4, rows=6):
        self.mamba_cache = type("MC", (), {})()
        self.mamba_cache.temporal = torch.arange(L * rows * 2 * 3, dtype=torch.float32).view(L, rows, 2, 3)
        self.mamba_cache.conv = [torch.arange(L * rows * 5 * 2, dtype=torch.float32).view(L, rows, 5, 2) + 1000]


def test_snapshot_rows_copies_the_rows_and_is_immune_to_later_writes():
    hp = _arena_mamba_host(pending_rows=(0, 1))
    dev = _Dev2()
    host = torch.tensor([3, 4])                 # arena rows 0, 1 (staging_rows=3), both pending
    didx = torch.tensor([5, 2])
    shim, idx, tok = hp.snapshot_rows(dev, host, didx)
    want_t = dev.mamba_cache.temporal[:, [5, 2]].clone()
    want_c = dev.mamba_cache.conv[0][:, [5, 2]].clone()
    dev.mamba_cache.temporal.add_(1.0)          # a later forward changes the live rows
    dev.mamba_cache.conv[0].mul_(0.0)
    assert torch.equal(shim.mamba_cache.temporal, want_t)
    assert torch.equal(shim.mamba_cache.conv[0], want_c)
    assert idx.tolist() == [0, 1]
    # the write path reads the shim like a device pool: row i of the op
    assert torch.equal(shim.mamba_cache.temporal[2].index_select(0, idx), want_t[2])


def test_snapshot_rows_refuses_what_the_fence_must_keep():
    dev = _Dev2()
    assert _arena_mamba_host(pending_rows=()).snapshot_rows(dev, torch.tensor([3]), torch.tensor([1])) is None
    assert _arena_mamba_host().snapshot_rows(dev, torch.tensor([1]), torch.tensor([1])) is None  # staging row
    hp = _arena_mamba_host()
    hp.arena = None
    assert hp.snapshot_rows(dev, torch.tensor([3]), torch.tensor([1])) is None


def test_pool_group_stages_only_mamba_transfers():
    from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup

    hp = _arena_mamba_host()
    dev = _Dev2()
    grp = object.__new__(HostPoolGroup)
    grp.entry_map = {PoolName.MAMBA: type("E", (), {"host_pool": hp, "device_pool": dev})()}
    t = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]), device_indices=torch.tensor([4]))
    out, toks = grp.snapshot_mamba_transfers([t])
    assert out[0]._snapshot_pool.mamba_cache.temporal.shape[1] == 1
    assert out[0].device_indices.tolist() == [0] and toks == [(hp, 0)]
    assert t.device_indices.tolist() == [4]     # the op's own transfer is not mutated
