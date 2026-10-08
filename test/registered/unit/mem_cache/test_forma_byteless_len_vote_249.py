"""#249 (rc12t, 8 W88 in 11 min, befund_w88_host_pool.md): on a Form A D
group the expert workers' KV host pools are byteless (0 B per row, JG) and,
under R12, keep every row until the reset -- the synced 353573 rows filled
while TP0's arena still had the span. The #580 vote took the worker's
allocated length as the group MIN: ``#915 PREFETCH TRUNCATED need=54336
got=49216`` on every rank, 64 requeues, ``W88 arm=host_pool_shortfall``.

Fix: a byteless pool's row is a placeholder, not capacity -- its id space
grows by whole pages (0 B) instead of refusing, so a worker's length vote is
its span again and the group's room is TP0's. Instrument: the group trim
names the rank whose own length set the MIN (``cut_rank=``), and the W88
line carries it (``min_rank=``).

Driven through the REAL ``UnifiedRadixCache.prefetch_from_storage`` on three
simulated ranks (the H99 harness) with real ``MHATokenToKVPoolHost``
bookkeeping on the workers."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_h99",
    os.path.join(os.path.dirname(__file__), "..", "managers", "test_nf_form_a_prefetch_span_h99.py"),
)
h99 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h99)

SPAN = 5440          # the host's span (rc12t: need=54336, scaled)
WORKER_ROWS = 4096   # the worker pool's synced size, every row held (R12)


def _byteless_pool(rows: int, used: int) -> MHATokenToKVPoolHost:
    """Real MHATokenToKVPoolHost bookkeeping, 0 kv-heads (the Form A worker)."""
    p = object.__new__(MHATokenToKVPoolHost)
    p.page_size = 1
    p.size = rows
    p.page_num = rows
    p.byteless = True
    p.lock = threading.RLock()
    p.layout = "page_first"
    p.layer_num, p.head_num, p.head_dim = 2, 0, 8
    p.dtype = torch.bfloat16
    p.device = "cpu"
    p.budget_label = "form-a-worker-kv"
    p.kv_buffer = torch.empty((2, rows, 2, 0, 8), dtype=p.dtype)
    p.clear()
    if used:
        assert p.alloc(used) is not None
    return p


class _Pool:
    """TP0's side: the arena has the room (or `room` rows of it)."""

    def __init__(self, room: int):
        self.room = room

    def alloc(self, n):
        if n > self.room:
            return None
        self.room -= n
        return torch.arange(n)

    def available_size(self):
        return self.room


def _intake(pools, rid="pdflip-7-31", spans=None):
    group = h99.MockGlooGroup()
    caches = {}
    spans = spans or {0: SPAN, 1: SPAN, 2: SPAN}

    def _rank(r):
        caches[r] = c = h99._carrier(r, group)
        c.cache_controller.mem_pool_host = pools[r]
        c._pdflip_rank_label = r
        c.prefetch_cut_terms = types.MethodType(UnifiedRadixCache.prefetch_cut_terms, c)
        c.prefetch_from_storage(rid, h99._host_node(), list(range(spans[r])), last_hash=None, prefix_keys=None)
        return h99._registered_len(c, rid)

    with h99._Env():
        results, errors = h99.run_ranks(_rank)
    return caches, results, errors, group


def test_rc12t_full_workers_no_longer_cut_the_group():
    """RED on e39b37d011: the workers' pools are full (available=0), their
    length vote is 0, the whole read is voted down -- nothing registers
    (rc12t: the partial 49216 of 54336, then W88). GREEN: every rank
    registers TP0's full span."""
    pools = {0: _Pool(1 << 20), 1: _byteless_pool(WORKER_ROWS, WORKER_ROWS),
             2: _byteless_pool(WORKER_ROWS, WORKER_ROWS)}
    _c, results, errors, group = _intake(pools)
    assert errors == {}, errors
    assert group.errors == []
    assert results == {0: (SPAN, SPAN), 1: (SPAN, SPAN), 2: (SPAN, SPAN)}, results


def test_a_part_full_worker_no_longer_truncates():
    """rc12t's own numbers scaled: the worker has 4096 of 5440 rows free-able
    -> the base trims every rank to the worker's room. GREEN: full span."""
    pools = {0: _Pool(1 << 20), 1: _byteless_pool(WORKER_ROWS, 0),
             2: _byteless_pool(WORKER_ROWS, 0)}
    _c, results, errors, _g = _intake(pools)
    assert errors == {}, errors
    assert set(results.values()) == {(SPAN, SPAN)}, results


def test_tp0_short_still_cuts_and_is_named(caplog):
    """TP0's own room still binds (the host is the one pool with bytes), and
    the trim names it on every rank: cut_rank=0, W88 terms min_rank=0."""
    pools = {0: _Pool(4000), 1: _byteless_pool(WORKER_ROWS, 0),
             2: _byteless_pool(WORKER_ROWS, 0)}
    with caplog.at_level(logging.WARNING):
        caches, results, errors, _g = _intake(pools, rid="pdflip-cut")
    assert errors == {}, errors
    assert set(results.values()) == {(4000, 4000)}, results
    lines = [r.getMessage() for r in caplog.records if "#915 PREFETCH TRUNCATED" in r.getMessage()]
    assert len(lines) == 3 and all("cut_rank=0 " in ln for ln in lines), lines
    assert {c.prefetch_cut_terms("pdflip-cut") for c in caches.values()} == {
        f"min_rank=0 group_len=4000 need={SPAN}"
    }
    assert caches[1].prefetch_cut_terms("never") == "min_rank=-"


def test_h99_worker_deeper_match_still_caps_the_group_by_its_key():
    """WHY THE WORKER KEEPS ITS LENGTH VOTE (and the pool grows instead of the
    worker abstaining in slot 2): a worker whose shadow anchor sits DEEPER
    than the host's has a SHORTER key -- it cannot register more tokens than
    that key holds. Its length vote still caps the group (uniform trim, the
    H99 case), and the trim names the worker (cut_rank=1). An abstention in
    slot 2 would let the group length exceed this rank's own rows: the
    ``group > local`` desync stop on the worker."""
    short = SPAN - 1000
    pools = {0: _Pool(1 << 20), 1: _byteless_pool(WORKER_ROWS, WORKER_ROWS),
             2: _byteless_pool(WORKER_ROWS, WORKER_ROWS)}
    caches, results, errors, _g = _intake(
        pools, rid="r-deep", spans={0: SPAN, 1: short, 2: SPAN})
    assert errors == {}, errors
    assert set(results.values()) == {(short, short)}, results
    assert all(
        c.prefetch_cut_terms("r-deep").startswith(f"min_rank=1 group_len={short} ")
        for c in caches.values()
    ), {r: c.prefetch_cut_terms("r-deep") for r, c in caches.items()}


def test_the_byteless_pool_grows_by_pages_and_clear_restores_it():
    p = _byteless_pool(64, 64)
    got = p.alloc(128)
    assert got is not None and got.tolist() == list(range(64, 192))
    assert p.size == 192 and p.kv_buffer.shape[1] == 192 and p.kv_buffer.numel() == 0
    p.free(got)                          # the double-free guard knows the grown ids
    assert p.available_size() == 128
    p.clear()
    assert p.size == 64 and p.available_size() == 64 and p.kv_buffer.shape[1] == 64


def test_a_pool_with_bytes_is_unchanged():
    """27B / TP0 / every classic pool: a full pool still refuses."""
    p = _byteless_pool(64, 64)
    p.byteless = False
    assert p.alloc(64) is None
    assert p.size == 64


def test_w88_names_the_min_rank():
    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._pdflip_store_load_terminal)
    assert "prefetch_cut_terms" in src
