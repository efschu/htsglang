"""#251c on a Form A group: the KV stage form acts on the attention host only.

rc12z11 (unified 590fa56a02, NF D boots 28.09. 09:01Z and 09:09Z) died twice
at the boot: TP1/TP2 (Form A expert workers, byteless KV) trimmed their QSA
compressed keys to S0 like TP0's KV -- ``kv_stage_born`` right after the
``torch.zeros`` that made them -- and faulted (GPU coredump, surfaced at the
next sync, ``KvRowCap._apply``, exit -6); TP0 died in the fill kernel of its
next buffer. Two holes, both closed here:

(1) A Form A worker trims nothing: ``kv_stage_trims_here`` is False, the QSA
    keys are not staged, ``kv_stage_born`` passes every tensor through, and
    ``_KV_BORN`` stays empty (the worker has no cells). Its allocator keeps
    the replicated top-stage rows and caps (page ids in step with TP0's) --
    index lists, no pages.
(2) On the host the trim waits for every kernel still writing the tensor:
    ``_sync_before_unmap`` runs before the unmap, never after.
"""
from __future__ import annotations

import os
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt import rank_role  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

G = 4096
PAGE = 16
STAGES = "64,128,192"


class _Plan:
    def __init__(self, workers):
        self._w = set(workers)

    def role_of(self, rank):
        return "worker" if rank in self._w else "host"

    def is_worker(self, rank):
        return rank in self._w


class FakeTms:
    available = True

    def __init__(self, order=None):
        self.allocs = {}
        self.calls = []
        self.order = order if order is not None else []

    def add(self, t):
        size = dsv.align_up(t.numel() * t.element_size(), G)
        self.allocs[t.data_ptr()] = size

    def info(self, ptr):
        size = self.allocs.get(int(ptr))
        return None if size is None else dsv.AllocInfo(size, size, size, True)

    def set_spans(self, ptr, spans, *, now):
        self.calls.append((int(ptr), tuple(spans), now))
        self.order.append("unmap")
        return 0


class _Ctx:
    def __init__(self, stack):
        self.stack = stack

    def __enter__(self):
        for s in self.stack:
            s.__enter__()
        return self

    def __exit__(self, *a):
        for s in reversed(self.stack):
            s.__exit__(*a)


def _armed(tms, rank):
    from flliper.srt.environ import envs

    rank_role.set_form_a_role_plan(_Plan(workers={1, 2}), rank=rank)
    return _Ctx([mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}),
                 envs.FLLIPER_OPT_PDFLIP_D_SEAT_VRAM.override(True),
                 envs.FLLIPER_PDFLIP_D_KV_STAGE_TOKENS.override(STAGES),
                 envs.FLLIPER_PDFLIP_D_KV_STAGE_ROWS.override(32),
                 envs.FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS.override(""),
                 mock.patch.object(dsv, "_TMS", tms),
                 mock.patch.object(dsv, "granule_for", lambda _d: G),
                 mock.patch.object(dsv, "_KV_BORN", [])])


@pytest.fixture(autouse=True)
def _restore_role_plan():
    """Every test leaves the installed Form A plan exactly as it found it
    (27B 28.09.: a plan left behind made later suites order-dependent)."""
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


def _born(tms):
    t = torch.zeros(192 + PAGE, 256, dtype=torch.int32)
    tms.add(t)
    return t


# ---- (1) the worker ------------------------------------------------------------------

def test_a_form_a_worker_trims_nothing():
    """RED ON 89a0f3f166: the worker's tensor was trimmed to S0 at birth."""
    tms = FakeTms()
    with _armed(tms, rank=1):
        assert dsv.stage_form() is not None
        assert dsv.kv_stage_trims_here(192) is False
        t = _born(tms)
        assert dsv.kv_stage_born(t, pool_size=192, page_size=PAGE, name="k0") is t
        q = torch.zeros(3, 52 * 64, dtype=torch.int32)
        tms.add(q)
        dsv.kv_stage_born(q, pool_size=192, page_size=PAGE, name="qsa_compressed",
                          tokens_per_slot=4, layers=3, slots=52)
        assert tms.calls == [] and dsv._KV_BORN == []


def test_the_workers_allocator_keeps_the_hosts_rows_and_caps():
    """Page ids in step with TP0's: the top stage's rows, S0's cap -- the
    rank role alone decides the worker, not a runner attribute that is never
    set (rc12z11 passed is_form_a_worker=False from every rank)."""
    from flliper.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    import types

    with _armed(FakeTms(), rank=1):
        assert dsv.kv_stage_pool_tokens(10) == 192  # no S0 budget on a worker
        a = PagedTokenToKVPoolAllocator(192, page_size=PAGE, dtype=torch.int64, device="cpu",
                                        kvcache=types.SimpleNamespace(), need_sort=False)
        assert dsv.kv_stage_boot_cap(a, PAGE) == 4
        assert a.available_size() == 4 * PAGE


def test_the_qsa_keys_are_staged_on_the_host_only():
    """qsa_kv_pool decides ``_staged`` through ``kv_stage_trims_here``."""
    import inspect

    from flliper.srt.mem_cache import qsa_kv_pool

    src = inspect.getsource(qsa_kv_pool)
    assert "_staged = _dsv.kv_stage_trims_here(int(size))" in src
    with _armed(FakeTms(), rank=0):
        assert dsv.kv_stage_trims_here(192) is True
        assert dsv.kv_stage_trims_here(128) is False  # not the top stage's pool
    with _armed(FakeTms(), rank=2):
        assert dsv.kv_stage_trims_here(192) is False


# ---- (2) the host ------------------------------------------------------------------

def test_the_host_waits_for_the_fill_before_a_page_goes():
    """RED ON 89a0f3f166: the unmap ran with torch.zeros' fill in flight."""
    order = []
    tms = FakeTms(order)
    with _armed(tms, rank=0), mock.patch.object(
            dsv, "_sync_before_unmap", lambda _t: order.append("sync"), create=True):
        t = _born(tms)
        dsv.kv_stage_born(t, pool_size=192, page_size=PAGE, name="k0")
        assert order == ["sync", "unmap"]
        assert [p for p, _ in dsv._KV_BORN] == [t.data_ptr()]


def test_the_sync_is_a_device_sync_for_a_cuda_tensor():
    fake = mock.MagicMock()
    fake.is_cuda = True
    with mock.patch("torch.cuda.synchronize") as sync:
        dsv._sync_before_unmap(fake)
        sync.assert_called_once_with(fake.device)
        dsv._sync_before_unmap(torch.zeros(2))
        sync.assert_called_once()


def test_a_classic_boot_trims_as_before():
    """No role plan installed: TP0 of a classic D trims as #251c did."""
    tms = FakeTms()
    with _armed(tms, rank=0):
        rank_role.set_form_a_role_plan(None, 0)
        t = _born(tms)
        dsv.kv_stage_born(t, pool_size=192, page_size=PAGE, name="k0")
        assert len(tms.calls) == 1 and tms.calls[0][2] is True


# ---- the boot line (rc12z13: nothing said whether a rank had trimmed) ----------------

class _MappedTms(FakeTms):
    def add(self, t):
        super().add(t)
        self.mapped = getattr(self, "mapped", {})
        self.mapped[t.data_ptr()] = self.allocs[t.data_ptr()]

    def info(self, ptr):
        size = self.allocs.get(int(ptr))
        if size is None:
            return None
        return dsv.AllocInfo(size, self.mapped.get(int(ptr), size), size, True)

    def set_spans(self, ptr, spans, *, now):
        self.mapped[int(ptr)] = sum(h - l for l, h in spans)
        return super().set_spans(ptr, spans, now=now)


def _alloc():
    import types

    from flliper.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    return PagedTokenToKVPoolAllocator(192, page_size=PAGE, dtype=torch.int64, device="cpu",
                                       kvcache=types.SimpleNamespace(), need_sort=False)


def test_every_rank_says_at_boot_whether_it_trimmed(caplog):
    import logging

    caplog.set_level(logging.INFO)
    for rank, trims in ((0, True), (1, False)):
        caplog.clear()
        tms = _MappedTms()
        with _armed(tms, rank=rank):
            dsv.kv_stage_born(_born(tms), pool_size=192, page_size=PAGE, name="k0")
            assert dsv.kv_stage_boot_cap(_alloc(), PAGE) == 4
        line = [m for m in caplog.messages if "#251c KV-STAGE" in m]
        assert len(line) == 1, caplog.messages
        assert "form=64/128/192" in line[0] and f"trims_here={trims}" in line[0]
        assert ("born=1" if trims else "born=0") in line[0]
        assert "cap_pages=4/12" in line[0]
        if trims:  # 80 rows of 1 KiB mapped of the 208-row VA (4 KiB granule)
            assert "mapped=0.1/0.2 MiB" in line[0]
