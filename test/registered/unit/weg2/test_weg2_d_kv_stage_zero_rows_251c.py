"""#251c on a real KV pool: the stage feeds safe_zero_rows, it never assigns it.

rc12z13 (7b2c6ee5ef) hung in its first P->D flip (09:34:47Z): TP0's
resume_memory_occupation raised ``AttributeError: property 'safe_zero_rows' of
'MHATokenToKVPool' object has no setter`` -- apply_stage (#251c (2)) assigned
the property #656 (65432cea6d) had made read-only. The stage is the backing of
a saver-trimmed pool (no VMM owner, no watermark), so the pool learns it through
``set_stage_backed_rows`` and both ``safe_zero_rows`` and
``_committed_row_bound`` report it:

* at birth: a pool born at the top stage on the trimming rank bounds itself at
  S0 + page; a Form A worker (no trim) and a pool off the form stay unbounded;
* at a wake: S0 -> S1 -> S0 moves the bound with the pages (the runtime test
  drives apply_stage on a real pool);
* a hybrid pool passes the bound to its full-attention pool;
* the idle flush zeroes up to the stage and not a row above it.
"""
import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

G = 4096
PAGE = 16
STAGES = "64,128,192"


class _AnyTms:
    """A saver that owns every tensor it is asked about (the pool allocates
    its own buffers, so their pointers are not known in advance)."""

    available = True

    def __init__(self):
        self.calls = []

    def info(self, ptr):
        return dsv.AllocInfo(1 << 30, 1 << 30, 1 << 30, True)

    def set_spans(self, ptr, spans, *, now):
        self.calls.append((int(ptr), tuple(spans), now))
        return 0


class _Plan:
    def __init__(self, workers):
        self._w = set(workers)

    def role_of(self, rank):
        return "worker" if rank in self._w else "host"

    def is_worker(self, rank):
        return rank in self._w


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


def _armed(tms, rank=None):
    from sglang.srt.environ import envs

    if rank is not None:
        rank_role.set_form_a_role_plan(_Plan({1, 2}), rank=rank)
    return _Ctx([mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}),
                 envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(True),
                 envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.override(STAGES),
                 envs.SGLANG_WEG2_D_KV_STAGE_ROWS.override(32),
                 envs.SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS.override(""),
                 mock.patch.object(dsv, "_TMS", tms),
                 mock.patch.object(dsv, "granule_for", lambda _d: G),
                 mock.patch.object(dsv, "_KV_BORN", [])])


def teardown_function(_fn):
    rank_role.set_form_a_role_plan(None, 0)


def _pool(size=192):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool(size, PAGE, torch.bfloat16, 1, 8, 2, "cpu", False,
                            enable_alt_stream=False)


def test_a_pool_born_at_the_top_stage_is_bounded_at_s0():
    """RED ON 4406e3b67d: born trimmed, but safe_zero_rows said 'all backed'."""
    tms = _AnyTms()
    with _armed(tms):
        pool = _pool()
    assert tms.calls  # its K and V buffers were trimmed at birth
    assert pool.safe_zero_rows == 64 + PAGE
    assert pool._committed_row_bound() == 64 + PAGE


def test_no_trim_no_bound():
    with _armed(_AnyTms(), rank=1):  # a Form A worker trims nothing
        worker = _pool()
    with _armed(_AnyTms()):
        small = _pool(size=128)  # not the top stage's pool
    plain = _pool()  # no form at all
    for pool in (worker, small, plain):
        assert pool.safe_zero_rows is None and pool._committed_row_bound() is None


def test_the_bound_moves_and_a_hybrid_pool_passes_it_down():
    """RED ON 4406e3b67d: there was no way to bound a pool but to assign the
    property, which raised AttributeError in the first wake."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    pool = _pool()
    for rows in (128 + PAGE, 64 + PAGE, 192 + PAGE):
        dsv.bound_stage_rows(pool, rows)
        assert pool.safe_zero_rows == rows == pool._committed_row_bound()
    full = _pool()
    hybrid = object.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = full
    dsv.bound_stage_rows(hybrid, 128 + PAGE)
    assert full.safe_zero_rows == 128 + PAGE
    dsv.bound_stage_rows(SimpleNamespace(), 80)  # no KV pool: nothing to bound


def test_the_flush_stops_at_the_stage():
    from sglang.srt.mem_cache.memory_pool import zero_kv_data_buffers

    pool = _pool()
    for t in pool.k_buffer + pool.v_buffer:
        t.fill_(1)
    dsv.bound_stage_rows(pool, 64 + PAGE)
    assert zero_kv_data_buffers(pool) == 4
    k = pool.k_buffer[0]
    assert float(k[: 64 + PAGE].abs().sum()) == 0.0
    assert float(k[64 + PAGE:].abs().min()) == 1.0  # rows above the stage untouched
