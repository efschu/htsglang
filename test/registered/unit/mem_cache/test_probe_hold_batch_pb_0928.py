"""PB (28.09.): the probe hold references its pages in ONE arena call.

27B rc12z24 P (dkr27browauthoritybar1w109281556, bb84760576) PP0: weg2-0-13's
store probe ended at 15:59:14 (``#1028B FETCH CAP ... claimed=45055``), its
``#257 PROBE-HOLD ... pages=45055 held=45055`` line came at 15:59:21, its read
took 50 ms: ``WEG2-LOAD-DEVICE ... queue_ms=7809 read_ms=50``. The queue was
``probe_hold.pin`` -- one ctypes ``ref_slots([slot], +1)`` per page, 45055 of
them, on the prefetch thread. Same shape for weg2-0-6 (held=17031,
queue_ms=5819) and weg2-0-5 (held=3022, 1255). Not the RO slots, not the
wake workers, not the write-behind (its passes in that window: 49-73 ms).

Pinned on the REAL C arena (temp file):
* ``ref_slots_mask_np`` gives exactly the per-slot verdict the one-at-a-time
  loop gave -- COMPLETE and CLAIMED take the reference, a FREE slot and -1
  refuse -- and moves each refcount by exactly one;
* ``pin`` holds the same pages with the same slots as the old loop, in one
  call (ledger recorded by name); an arena without the batch keeps the loop;
* 20000 pages pin at least 10x faster than the per-page loop (hermetic bar).
"""

import os
import shutil
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache import probe_hold  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

PAGE = 64


def _arena(tmp_path, slots):
    return ShmArena(str(tmp_path / "kv.bin"), PAGE, slots)


def _write(arena, stems):
    pay = torch.zeros((PAGE,), dtype=torch.uint8)
    got = arena.write(list(stems), [PAGE] * len(stems), [((0, PAGE),)] * len(stems),
                      [pay.data_ptr()] * len(stems))
    assert got == [1] * len(stems)


def _refcounts(arena, slots):
    """Each slot's refcount, read back through a -1 per slot (and restored)."""
    out = []
    for s in slots:
        n = 0
        while arena.ref_slots([int(s)], -1) == 1:
            n += 1
        for _ in range(n):
            arena.ref_slots([int(s)], +1)
        out.append(n)
    return out


@pytest.fixture(autouse=True)
def _clean():
    probe_hold._held_total = 0
    yield
    probe_hold._held_total = 0


def test_the_mask_is_the_per_slot_verdict(tmp_path):
    a = _arena(tmp_path, 16)
    stems = ["s%02d" % i for i in range(6)]
    _write(a, stems)
    fs, st = a.find_slots_np(stems)
    assert (st == 2).all()
    free_slot = int(max(fs)) + 1                       # never written: FREE
    batch = np.array(list(fs) + [free_slot, -1], dtype=np.int64)
    ok = a.ref_slots_mask_np(batch)
    assert ok.tolist() == [1] * 6 + [0, 0]
    assert _refcounts(a, fs) == [1] * 6


def test_pin_holds_exactly_what_the_loop_held(tmp_path):
    a = _arena(tmp_path, 64)
    stems = ["p%03d" % i for i in range(40)]
    _write(a, stems[:30])                              # 10 absent from the arena
    pool = types.SimpleNamespace(arena=a, arena_read=True)
    op1 = types.SimpleNamespace(request_id="batch")
    held1 = probe_hold.pin(op1, pool, stems)
    pins1 = op1.probe_pins.copy()                      # release sets them to -1
    probe_hold.release(op1, pool, 0)
    # the old loop: an arena proxy without the batch method
    proxy = types.SimpleNamespace(find_slots_np=a.find_slots_np, ref_slots=a.ref_slots, slots=a.slots)
    op2 = types.SimpleNamespace(request_id="loop")
    held2 = probe_hold.pin(op2, types.SimpleNamespace(arena=proxy, arena_read=True), stems)
    assert held1 == held2 == 30
    assert pins1.tolist() == op2.probe_pins.tolist()
    assert (pins1[:30] >= 0).all() and (pins1[30:] == -1).all()
    probe_hold.release(op2, types.SimpleNamespace(arena=proxy, arena_read=True), 0)
    fs, _st = a.find_slots_np(stems[:30])
    assert _refcounts(a, fs) == [0] * 30               # every hold given back


def test_twenty_thousand_pages_pin_in_one_call(tmp_path):
    n = 20000
    a = _arena(tmp_path, 2 * n + 8)   # the hold may take half the arena
    stems = ["q%05d" % i for i in range(n)]
    for i in range(0, n, 2000):
        _write(a, stems[i:i + 2000])
    pool = types.SimpleNamespace(arena=a, arena_read=True)
    t = time.perf_counter()
    op = types.SimpleNamespace(request_id="fast")
    assert probe_hold.pin(op, pool, stems) == n
    fast = time.perf_counter() - t
    probe_hold.release(op, pool, 0)
    proxy = types.SimpleNamespace(find_slots_np=a.find_slots_np, ref_slots=a.ref_slots, slots=a.slots)
    t = time.perf_counter()
    op2 = types.SimpleNamespace(request_id="slow")
    assert probe_hold.pin(op2, types.SimpleNamespace(arena=proxy, arena_read=True), stems) == n
    slow = time.perf_counter() - t
    probe_hold.release(op2, types.SimpleNamespace(arena=proxy, arena_read=True), 0)
    print("PB pin 20000 pages: batch %.1f ms, per-page loop %.1f ms (x%.0f)"
          % (fast * 1e3, slow * 1e3, slow / fast))
    assert slow / fast >= 10
