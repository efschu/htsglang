# SPDX-License-Identifier: Apache-2.0
"""L15-11c (AP D2, 01.10.): adapter binding + the D sleep-flush retain hook.

Part 1 (adapter): ``torch_memory_saver_adapter.set_keep_spans`` converts
row ranges to byte ranges relative to the allocation base
(``row * stride(0) * element_size()``) and calls the patched hook's C entry
``tms_set_keep_spans`` (tms_csrc/entrypoint.cpp:174).  A missing symbol and
the no-op adapter both answer -100, the caller's "this saver cannot keep".

Part 2 (added with the scheduler commit): a source check that the D sleep
flush calls ``l15_retain.retain_at_sleep`` only behind
``l15_plan.master_on`` and skips exactly the four destructive calls inside
the retained branch.
"""
import torch

from sglang.srt.utils import torch_memory_saver_adapter as tmsmod


class _RecordingSymbol:
    """Stands in for the C tms_set_keep_spans; records the ctypes args."""

    def __init__(self, ret=7):
        self.calls = []
        self.ret = ret

    def __call__(self, ptr, n, lo, hi):
        self.calls.append(
            (
                ptr.value,
                n,
                [lo[i] for i in range(n)],
                [hi[i] for i in range(n)],
            )
        )
        return self.ret


def _patch_symbol(monkeypatch, fn):
    monkeypatch.setattr(
        tmsmod,
        "_weg2_ring_symbol",
        lambda name: fn if name == "tms_set_keep_spans" else None,
    )


def test_set_keep_spans_rows_to_bytes(monkeypatch):
    rec = _RecordingSymbol()
    _patch_symbol(monkeypatch, rec)
    # (10, 4) float32: one row = stride(0) * element_size = 4 * 4 = 16 bytes.
    t = torch.zeros(10, 4, dtype=torch.float32)
    rc = tmsmod._TorchMemorySaverAdapterReal().set_keep_spans(t, [(0, 3)])
    assert rc == 7, "the adapter returns the C code"
    assert len(rec.calls) == 1
    ptr, n, lo, hi = rec.calls[0]
    assert ptr == t.data_ptr(), "the allocation base is the tensor data_ptr"
    assert n == 1
    assert lo == [0] and hi == [48], "rows (0, 3) -> bytes [0, 48)"
    # Unsorted multi-range input is normalised and scaled the same way.
    tmsmod._TorchMemorySaverAdapterReal().set_keep_spans(t, [(2, 5), (0, 1)])
    _, n2, lo2, hi2 = rec.calls[1]
    assert n2 == 2
    assert lo2 == [0, 32] and hi2 == [16, 80], "sorted, row*16 bytes"


def test_set_keep_spans_missing_symbol_is_minus_100(monkeypatch):
    monkeypatch.setattr(tmsmod, "_weg2_ring_symbol", lambda name: None)
    t = torch.zeros(4, 2, dtype=torch.float32)
    assert tmsmod._TorchMemorySaverAdapterReal().set_keep_spans(t, [(0, 2)]) == -100


def test_noop_adapter_set_keep_spans_is_minus_100():
    noop = tmsmod.TorchMemorySaverAdapter.create(enable=False)
    t = torch.zeros(4, 2, dtype=torch.float32)
    assert noop.set_keep_spans(t, [(0, 2)]) == -100
