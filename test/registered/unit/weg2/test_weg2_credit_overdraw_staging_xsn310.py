"""Task #20 (18.09.): an on-card IPC staging may take FREE, UNPROMISED
VRAM at the start of a leg (overdraw); the waker's counter balance is
untouched; refund gives it back."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.weg2_memory_saver import MIB, VramCredit  # noqa: E402


def _credit(tmp_path, name):
    c = VramCredit(name, credit_dir=str(tmp_path))
    c.publish("weights_0", 0)  # a leg is open, nothing published yet
    return c


def test_overdraw_takes_only_free_minus_floor_minus_promised(tmp_path):
    c = _credit(tmp_path, "GPU-xsn310-a")
    # nothing published: the counter path refuses ...
    assert not c.debit("ipc-stage", 1309 * MIB)
    # ... free VRAM covers it (11920 MiB free, floor 767, nothing promised)
    assert c.debit("ipc-stage", 1309 * MIB, free_bytes=11920 * MIB, floor_bytes=767 * MIB)
    # a second staging must respect what is already staged: 11920-767-1309 = 9844 >= 1309
    assert c.debit("ipc-stage", 1309 * MIB, free_bytes=11920 * MIB, floor_bytes=767 * MIB)
    # (11920 - 767) / 1309 = 8.5: eight stagings fit, the ninth is refused
    for _ in range(6):
        assert c.debit("ipc-stage", 1309 * MIB, free_bytes=11920 * MIB, floor_bytes=767 * MIB)
    assert not c.debit("ipc-stage", 1309 * MIB, free_bytes=11920 * MIB, floor_bytes=767 * MIB)


def test_overdraw_never_touches_the_wakers_balance_and_refund_returns_it(tmp_path):
    c = _credit(tmp_path, "GPU-xsn310-b")
    c.publish("weights_1", 2000 * MIB)  # the waker is promised 2000
    # overdraw beyond the promise: free 6000, floor 0, promised 2000 -> 4000 stageable
    assert c.debit("ipc-stage", 3000 * MIB, free_bytes=6000 * MIB, floor_bytes=0)
    # the counter balance (2000) is still fully available to a counter debit
    assert c.debit("ipc-stage", 2000 * MIB)
    assert not c.debit("ipc-stage", 1 * MIB)
    # refund of the overdrawn staging frees the overdraw first, not the counter
    c.refund("ipc-stage", 3000 * MIB)
    assert not c.debit("ipc-stage", 1 * MIB)  # counter still spent
    # ... and the freed overdraw can be staged again (free 6000 - staged 2000 >= 3000)
    assert c.debit("ipc-stage", 3000 * MIB, free_bytes=6000 * MIB, floor_bytes=0)
    assert not c.debit("ipc-stage", 1001 * MIB, free_bytes=6000 * MIB, floor_bytes=0)


def test_the_stage_charge_passes_free_and_floor():
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = open(wu.__file__).read()
    i = src.index("def _charge(nbytes: int):")
    blk = src[i:i + 900]
    assert "free_bytes=_free, floor_bytes=_floor" in blk
    assert "_weg2_free_bytes()" in blk and "_weg2_corridor_floor_bytes()" in blk
