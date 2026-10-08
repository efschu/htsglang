"""DP-NACHLAUF 02.10.: the wake restore's re-zero through a wide view.

Every D wake of N5m/N5p/N5q: 'PDFLIP-WAKE-INVARIANT ... re-zeroed ... in
85-95 ms' on TP0 (5090) and 169-189 ms on TP1/TP2 (3080) for ~8-10 GB per
rank -- the fill ran element-wise over fp8 KV buffers and the uint8 mamba
envelope, one byte per element. Pinned (red before): a contiguous buffer is
zeroed through an int64 view of the same bytes (identical result, counted
wide); strided / odd-sized / misaligned buffers keep the plain zero_() and are
counted narrow; the switch off is the plain path everywhere;
zero_kv_data_buffers honours safe_zero_rows exactly as before; the flush line
names wide/narrow/bytes/ms; the mamba envelope goes through the wide zero.
"""
import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache import memory_pool as mp  # noqa: E402


def _dirty(shape, dtype=torch.uint8):
    t = torch.empty(shape, dtype=torch.uint8)
    t.fill_(0x7F)                              # 0x7f = fp8 e4m3 NaN
    return t.view(dtype) if dtype != torch.uint8 else t


def test_contiguous_fp8_goes_wide_and_is_all_zero(monkeypatch):
    monkeypatch.delenv(mp.ZERO_WIDE_ENV, raising=False)
    mp.zero_wide_stats_take()
    t = _dirty((64, 8, 16), torch.float8_e4m3fn)
    assert mp.zero_wide_(t) is True
    assert int(t.view(torch.uint8).sum()) == 0
    st = mp.zero_wide_stats_take()
    assert st == {"wide": 1, "narrow": 0, "bytes": 64 * 8 * 16}


def test_strided_odd_and_switch_off_stay_narrow(monkeypatch):
    monkeypatch.delenv(mp.ZERO_WIDE_ENV, raising=False)
    base = _dirty((64, 16))
    strided = base[:, ::2]
    assert mp.zero_wide_(strided) is False
    assert int(strided.sum()) == 0 and int(base[:, 1::2].min()) == 0x7F   # only the view
    odd = _dirty((7,))
    assert mp.zero_wide_(odd) is False and int(odd.sum()) == 0
    monkeypatch.setenv(mp.ZERO_WIDE_ENV, "0")
    t = _dirty((64, 16))
    assert mp.zero_wide_(t) is False and int(t.sum()) == 0


def test_a_slice_of_rows_is_wide_and_leaves_the_rest(monkeypatch):
    monkeypatch.delenv(mp.ZERO_WIDE_ENV, raising=False)
    t = _dirty((100, 32))
    assert mp.zero_wide_(t[:40]) is True
    assert int(t[:40].sum()) == 0 and int(t[40:].min()) == 0x7F


def test_zero_kv_data_buffers_keeps_safe_zero_rows(monkeypatch):
    monkeypatch.delenv(mp.ZERO_WIDE_ENV, raising=False)
    k = [_dirty((50, 4, 8), torch.float8_e4m3fn) for _ in range(3)]
    v = [_dirty((50, 4, 8), torch.float8_e4m3fn) for _ in range(3)]
    pool = SimpleNamespace(k_buffer=k, v_buffer=v, safe_zero_rows=20)
    mp.zero_wide_stats_take()
    assert mp.zero_kv_data_buffers(pool) == 6
    for t in k + v:
        u = t.view(torch.uint8)
        assert int(u[:20].sum()) == 0 and int(u[20:].min()) == 0x7F
    assert mp.zero_wide_stats_take()["wide"] == 6


def test_flush_line_and_envelope_use_the_wide_zero():
    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._flush_zero_kv_buffers)
    assert "wide=%d narrow=%d bytes=%d ms=%.1f" in src
    assert "flush: zeroed %d KV data buffers (%d unbacked layout(s) skipped)" in src
    rs = inspect.getsource(mp.MambaPool.reset_state) if hasattr(mp, "MambaPool") else inspect.getsource(mp)
    assert "zero_wide_(raw)" in rs
