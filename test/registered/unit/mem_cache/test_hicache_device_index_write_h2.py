"""H2 (23.09.): a direct-backend HiCache write op keeps its device indices on
the card when every pool of the op can take them.

The defect (boots fnFL2x80..x87, group P, ``--hicache-io-backend direct
--hicache-mem-layout layer_first``): ``HybridCacheController.start_writing``
normalised every write op through ``move_hybrid_indices`` ->
``move_indices``, whose direct branch is ``device_indices.cpu()`` on the
COMPUTE stream. The scheduler thread therefore waited for the card's queued
forward inside every chunk publish: ``WEG2 CHUNK-PUBLISH ... 'ms'`` 544-1956
for the second chunk against 25-63 for a publish with nothing in flight --
although the KV/mamba arena pools and the QSA sidecar pool all write from
device indices. Hermetic: no CUDA; the "on card" indices are a stand-in that
only answers ``is_cuda``.
"""

import os
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_write_path
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool, HostPoolGroup
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CardIndices:
    """Device indices as the write path sees them on the rig: on the card.
    Reading their VALUES on the host is the sync under test, so any attempt
    to do so fails loudly."""

    is_cuda = True

    def numel(self):
        return 4

    def cpu(self):
        raise AssertionError("device indices were read on the host (the H2 sync)")


class _FakeEvent:
    def record(self):
        pass

    def wait(self, stream):
        pass


class _BusyStream:
    def query(self):
        return False   # the forward is still in flight


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    @contextmanager
    def stream(stream):
        yield

    @staticmethod
    def current_stream():
        return _BusyStream()


class _Group:
    def __init__(self, refusal):
        self.refusal = refusal
        self.calls = []
        self.layout = "layer_first"

    def backup_accepts_device_indices(self, host_indices, device_indices, pool_transfers=None):
        return self.refusal

    def backup_from_device_indices(self, device_pool, host_indices, device_indices, pool_transfers=None):
        self.calls.append(("device", host_indices, device_indices, pool_transfers))

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend,
                                     pool_transfers=None):
        self.calls.append(("normalised", host_indices, device_indices, io_backend))


def _controller(refusal):
    op = CacheOperation(
        host_indices=torch.arange(100, 104, dtype=torch.int64),
        device_indices=_CardIndices(),
        node_id=7,
        pool_transfers=[PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([9]),
                                     device_indices=_CardIndices())],
    )
    c = HybridCacheController.__new__(HybridCacheController)
    c.write_queue = [op]
    c.io_backend = "direct"
    c.mem_pool_host = _Group(refusal)
    c.mem_pool_device = None
    c.has_draft = False
    c._dcp_owner_ctx_cache = None          # uneven-DCP owner rule off (PP phase)
    c.write_stream = object()
    c.ack_write_queue = []
    c._record_transfer_indices_on_stream = lambda *a: None
    moved_host = torch.arange(100, 104, dtype=torch.int64)
    c.move_hybrid_indices = mock.Mock(return_value=(moved_host, torch.arange(4), op.pool_transfers))
    return c, op


def _run(c):
    before = hicache_write_path.STATS.copy()
    with mock.patch.object(hybrid_cache_controller, "device_module", _FakeDeviceModule), \
            mock.patch.object(hybrid_cache_controller, "consume_gate", lambda *a: True):
        c.start_writing()
    return hicache_write_path.STATS.since(before)


def test_direct_backend_op_skips_the_host_normalisation():
    """Regression (x80-x87): on the pre-fix code start_writing calls
    move_hybrid_indices for every direct-backend op -- the device-index
    D2H that blocks on the forward in flight."""
    c, op = _controller(refusal="")
    c.move_hybrid_indices.side_effect = AssertionError("normalised an op every pool accepted")
    with envs.SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE.override(True):
        delta = _run(c)
    c.move_hybrid_indices.assert_not_called()
    [(kind, host, dev, transfers)] = c.mem_pool_host.calls
    assert kind == "device"
    assert host is op.host_indices and dev is op.device_indices
    assert transfers is op.pool_transfers           # extra pools unmoved too
    assert len(c.ack_write_queue) == 1 and c.ack_write_queue[0].node_ids == op.node_ids
    assert (delta.device_ops, delta.fallback_ops, delta.busy_at_entry) == (1, 0, 1)


def test_a_refusing_pool_and_the_kill_switch_keep_the_normalisation():
    """Negative branches: a pool that cannot take card indices (a staging
    row, an unbound arena) and SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE=0 both
    take the old path, with its io backend, counted by reason."""
    for env_on, refusal, reason in ((True, "kv:staging_rows", "kv:staging_rows"),
                                    (False, "", "off")):
        c, _ = _controller(refusal=refusal)
        with envs.SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE.override(env_on):
            delta = _run(c)
        c.move_hybrid_indices.assert_called_once()
        [(kind, _h, _d, io_backend)] = c.mem_pool_host.calls
        assert (kind, io_backend) == ("normalised", "direct")
        assert delta.fallback_reasons == {reason: 1} and delta.device_ops == 0


def test_draft_and_dcp_veto_the_card_form():
    """The draft pool's backup and the uneven-DCP owner rule both need the
    normalised pairs; either armed must refuse even when every pool accepts."""
    c, op = _controller(refusal="")
    with envs.SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE.override(True):
        c.draft_tier_armed = lambda direction: True
        assert c._device_index_write_refusal(op) == "draft"
        c.draft_tier_armed = lambda direction: False
        c._dcp_owner_ctx_cache = (3, 0, 1)
        assert c._device_index_write_refusal(op) == "dcp"
        c._dcp_owner_ctx_cache = None
        assert c._device_index_write_refusal(op) == ""
        c.io_backend = "kernel"   # no D2H to remove there
        assert c._device_index_write_refusal(op) == "io_backend"


def test_group_asks_the_anchor_and_every_extra_pool():
    """One refusing extra pool (or an unbound/unresolved transfer) must refuse
    the whole op -- a half-moved op would write one pool from wrong rows."""
    ok = SimpleNamespace(backup_accepts_device_indices=lambda h, d: "")
    no = SimpleNamespace(backup_accepts_device_indices=lambda h, d: "qsa_indexer:layout")
    g = HostPoolGroup.__new__(HostPoolGroup)
    g.anchor_entry = SimpleNamespace(host_pool=ok)
    g.entry_map = {PoolName.MAMBA: SimpleNamespace(host_pool=ok),
                   PoolName.QSA_INDEXER: SimpleNamespace(host_pool=no)}
    h, d = torch.arange(4), torch.arange(4)
    mamba = PoolTransfer(name=PoolName.MAMBA, host_indices=h, device_indices=d)
    qsa = PoolTransfer(name=PoolName.QSA_INDEXER, host_indices=h, device_indices=d)
    assert g.backup_accepts_device_indices(h, d, [mamba]) == ""
    assert g.backup_accepts_device_indices(h, d, [mamba, qsa]) == "qsa_indexer:layout"
    assert g.backup_accepts_device_indices(
        h, d, [PoolTransfer(name=PoolName.MAMBA, host_indices=None, device_indices=d)]
    ).startswith("unresolved:")
    g.entry_map.pop(PoolName.MAMBA)
    assert g.backup_accepts_device_indices(h, d, [mamba]).startswith("unbound:")
    g.anchor_entry = SimpleNamespace(host_pool=no)
    assert g.backup_accepts_device_indices(h, d, []) == "qsa_indexer:layout"


def test_arena_pools_accept_only_all_arena_host_rows():
    """Boundary: host ids [S, S + A*P) are arena rows; one staging row (< S)
    or one past the end would reach the base pool's backend branch, which
    reads host indices -- refuse."""
    S, A, P = 10, 2, 4
    kv = SimpleNamespace(arena=object(), staging_rows=S, arena_tokens=A * P)
    acc = lambda pool, ids: ArenaMHAHostPool.backup_accepts_device_indices(   # noqa: E731
        pool, torch.tensor(ids, dtype=torch.int64), None)
    assert acc(kv, list(range(S, S + A * P))) == ""
    assert acc(kv, [S - 1, S]) == "kv:staging_rows"
    assert acc(kv, [S + A * P]) == "kv:staging_rows"
    assert acc(SimpleNamespace(arena=None), [S]) == "kv:arena_unbound"

    mp = SimpleNamespace(arena=object(), staging_rows=S, arena_slots=A)
    mp._split = types.MethodType(ArenaMambaPoolHost._split, mp)
    macc = lambda ids: ArenaMambaPoolHost.backup_accepts_device_indices(   # noqa: E731
        mp, torch.tensor(ids, dtype=torch.int64), None)
    assert macc([S, S + A - 1]) == ""
    assert macc([S + A]) == "mamba:staging_rows"


def test_qsa_pool_accepts_whole_pages_into_layer_first_only():
    """The card form runs the kernel/layer_first page-row copy: whole pages
    only (a partial page is the token-granular helper's), layer_first only
    (the host pointers exist only there), pinned host memory only (the
    kernel stores through the device mapping)."""
    pool = SimpleNamespace(layout="layer_first", data_ptrs=object(), slot_page_size=64,
                           pool_name="qsa_indexer", pin_memory=True)
    acc = lambda h, d: DeepSeekV4PagedHostPool.backup_accepts_device_indices(pool, h, d)   # noqa: E731
    with mock.patch.object(memory_pool_host, "transfer_kv_all_layer_mla", object()):
        assert acc(torch.arange(128), torch.arange(128)) == ""
        assert acc(torch.arange(96), torch.arange(96)) == "qsa_indexer:partial_page"
        assert acc(torch.arange(128), torch.arange(64)) == "qsa_indexer:partial_page"
        pool.layout = "page_first"
        assert acc(torch.arange(128), torch.arange(128)) == "qsa_indexer:layout"
        pool.layout, pool.pin_memory = "layer_first", False
        assert acc(torch.arange(128), torch.arange(128)) == "qsa_indexer:unpinned"
    pool.pin_memory = True
    with mock.patch.object(memory_pool_host, "transfer_kv_all_layer_mla", None):
        assert acc(torch.arange(128), torch.arange(128)) == "qsa_indexer:no_kernel"


def test_publish_line_carries_the_fields_the_boot_is_read_by():
    """The operator's boot instruction greps these names; blocked is wall
    minus thread CPU and never negative."""
    delta = hicache_write_path.WritePathStats(ops=2, device_ops=1, fallback_ops=1,
                                              fallback_reasons={"draft": 1}, move_wall_ms=3.0)
    line = hicache_write_path.format_publish_line(
        n=5, wall_ms=40.0, cpu_ms=45.0, delta=delta,
        tally=hicache_write_path.PublishTally(n=5, wall_ms=900.0, cpu_ms=100.0))
    for field in ("H2-WRITE-PATH chunk n=5", "wall_ms=40.0", "cpu_ms=45.0", "blocked_ms=0.0",
                  "device_ops=1", "fallback=[draft:1]", "move_ms=3.0", "cum n=5",
                  "blocked_ms=800"):
        assert field in line, field
