"""W109b (rc12z4, 28.09. 06:05:47Z, flip 22, D->P wake): the credit cycle
that killed the group is resolved by the blocked DEPOSITOR, not named and died.

The measured form (P log 76100-76288, D log 113611-113618):
  waker2 (P PP2, card 2) waits for weights_15 (need 3202 MiB, free 1657);
  its credit is funded by sleeper2 (D TP2), which is blocked depositing
  weights_11 on lane p5 (2->1) to waker1;
  waker1 (P PP1) waits for weights_11 (need 2854 MiB, free 2725); its credit
  is funded by sleeper1 (D TP1), blocked depositing weights_15 on lane p3
  (1->2) to waker2.
Nobody collects, so nobody frees a ring slot, so nobody pauses.

The fix: the blocked depositor sees its own edge in the cycle, copies the
rest of the tag to host memory, returns (the caller pauses the tag, its VRAM
funds the waker) and a backlog feeds the ring from host when the collector
drains. These tests drive sleeper2's lane p5 on host memory (fake ops)."""
from __future__ import annotations

import ctypes
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))   # p3 = 1->2, p5 = 2->1
NONCE = "n1"
SLOT, RING = 4096, 2


class _HostOps:
    def create_stream(self, device):
        return 0

    def synchronize(self, stream):
        pass

    def memcpy_async(self, dst, src, n, stream):
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _lanes(tmp_path, group, rank):
    return b1.Bar1Lanes(NONCE, group, rank, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))


def _rc12z4_cycle(tmp_path):
    """Both P wakers in their credit wait for a while, sleeper1 already
    blocked on p3 (1->2). Sleeper2's own p5 edge is posted by its real wait."""
    root = str(tmp_path)
    old = time.time() - 10.0
    cd = os.path.join(root, f"weg2-bar1-{NONCE}", "credit")
    b1.post_flag(cd, "wait", "P", 1, {"tag": "weights_11", "since": old, "submitted": []})
    b1.post_flag(cd, "wait", "P", 2, {"tag": "weights_15", "since": old, "submitted": []})
    b1.post_flag(b1.flag_dir(NONCE, "p3", "P", root), "blocked", "22-weights_15", 0,
                 {"since": old, "wait": "free", "lane": "p3", "rank": 1})


def _pair(tmp_path):
    window = bytearray(SLOT * RING)
    dep = _lanes(tmp_path, "D", 2)            # sleeper2 = D TP2
    col = _lanes(tmp_path, "P", 1)            # waker1 = P PP1
    col.recv["p5"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=RING)
    dep.peers["p5"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=RING)
    return dep, col, window


def _tag(nbytes, name):
    src = bytearray(os.urandom(nbytes))
    dst = bytearray(nbytes)
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=nbytes, src_off=0, dst_off=0, param_name=name,
                             tag="weights_11", src_ptr=_addr(src), dst_ptr=_addr(dst))]
    return src, dst, descs


def _collect(col, descs, seq, out, key):
    out[key] = b1.run_bar1_units(descs, _HostOps(), lanes=col, lane_key="p5", role="dst",
                                 seq=seq, phase="collect", budget_s=10.0, log=lambda *_a: None)


def test_rc12z4_cycle_the_blocked_depositor_spills_and_returns(tmp_path):
    _rc12z4_cycle(tmp_path)
    dep, col, _w = _pair(tmp_path)
    nbytes = SLOT * 5 + 777                      # 6 batches > ring 2: the deposit blocks
    src, dst, descs = _tag(nbytes, "model.layers.29.mlp.experts.w13")
    want = bytes(src)
    t0 = time.perf_counter()
    why = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p5", role="src",
                            seq="22-weights_11", phase="deposit", budget_s=8.0,
                            log=lambda *_a: None)
    took = time.perf_counter() - t0
    # base: the wait ran its whole budget and refused ("no 'free' for batch 0")
    assert why == "", why
    assert took < 5.0, took
    # sleeper2's edge left the flags: the waker's cycle reader no longer closes it
    assert b1.credit_cycle(2, dep.read_credit_waits("P"), dep.read_blocked("P"), 1.0,
                           time.time()) is None
    # the caller pauses the tag now: its device bytes are gone
    src[:] = bytes(nbytes)
    # the waker's credit arrives, it collects -- every byte from the backlog
    out = {}
    th = threading.Thread(target=_collect, args=(col, descs, "22-weights_11", out, "c"))
    th.start()
    th.join(15)
    assert out.get("c") == ""
    assert bytes(dst) == want
    assert dep.join_backlog(10.0) == ""
    assert not dep.backlog_busy("p5")


def test_a_later_tag_on_a_spilled_lane_queues_behind_it_in_order(tmp_path):
    _rc12z4_cycle(tmp_path)
    dep, col, _w = _pair(tmp_path)
    src1, dst1, d1 = _tag(SLOT * 4 + 5, "a")
    src2, dst2, d2 = _tag(SLOT * 3 + 9, "b")
    want1, want2 = bytes(src1), bytes(src2)
    assert b1.run_bar1_units(d1, _HostOps(), lanes=dep, lane_key="p5", role="src",
                             seq="22-weights_11", phase="deposit", budget_s=8.0,
                             log=lambda *_a: None) == ""
    t0 = time.perf_counter()
    # the lane still belongs to the backlog: the next tag goes behind it at once
    assert b1.run_bar1_units(d2, _HostOps(), lanes=dep, lane_key="p5", role="src",
                             seq="22-weights_3", phase="deposit", budget_s=8.0,
                             log=lambda *_a: None) == ""
    assert time.perf_counter() - t0 < 1.0
    src1[:] = bytes(len(src1))
    src2[:] = bytes(len(src2))
    out = {}

    def _both():
        _collect(col, d1, "22-weights_11", out, "c1")
        _collect(col, d2, "22-weights_3", out, "c2")

    th = threading.Thread(target=_both)
    th.start()
    th.join(20)
    assert out.get("c1") == "" and out.get("c2") == ""
    assert bytes(dst1) == want1 and bytes(dst2) == want2
    assert dep.join_backlog(10.0) == ""


def test_no_cycle_no_spill_the_deposit_waits_for_its_collector(tmp_path):
    # only waker1 waits; nobody else is blocked: not a cycle
    old = time.time() - 10.0
    b1.post_flag(os.path.join(str(tmp_path), f"weg2-bar1-{NONCE}", "credit"), "wait", "P", 1,
                 {"tag": "weights_11", "since": old, "submitted": []})
    dep, col, _w = _pair(tmp_path)
    src, dst, descs = _tag(SLOT * 5 + 1, "c")
    t0 = time.perf_counter()
    why = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p5", role="src",
                            seq="22-weights_11", phase="deposit", budget_s=2.5,
                            log=lambda *_a: None)
    assert "no 'free'" in why and time.perf_counter() - t0 >= 2.4
    assert not dep.backlog_busy("p5")


def test_deposit_cycle_names_only_a_cycle_through_this_sleeper(tmp_path):
    _rc12z4_cycle(tmp_path)
    old = time.time() - 10.0
    root = str(tmp_path)
    s2 = _lanes(tmp_path, "D", 2)
    s0 = _lanes(tmp_path, "D", 0)
    assert s2.deposit_cycle() is None            # its own edge is not posted yet
    b1.post_flag(b1.flag_dir(NONCE, "p5", "P", root), "blocked", "22-weights_11", 3,
                 {"since": old, "wait": "free", "lane": "p5", "rank": 2})
    chain = s2.deposit_cycle()
    assert chain is not None and {(s, d) for s, d, _t in chain} == {(2, 1), (1, 2)}
    assert s0.deposit_cycle() is None            # sleeper0 is not an edge of it


# -- the waker's half: W109 only for a cycle that outlives the spill ---------

_CHAIN = [(2, 1, "weights_11"), (1, 2, "weights_15")]


def test_waker_names_a_cycle_that_outlives_the_depositors_answer(tmp_path, monkeypatch):
    import pytest

    from sglang.srt.managers.weg2_memory_saver import (
        VramCredit,
        Weg2XchgCreditCycleRefused,
    )

    monkeypatch.setenv(b1.ENV_CYCLE_GRACE_S, "1")
    credit = VramCredit("GPU-w109b-card2", credit_dir=str(tmp_path / "credit"))
    t0 = time.monotonic()
    with pytest.raises(Weg2XchgCreditCycleRefused):
        credit.wait_for(3202 * 1024 * 1024, budget_s=20.0, tag="weights_15",
                        free_bytes_now=0, cycle_reader=lambda: list(_CHAIN))
    took = time.monotonic() - t0
    # base: refused at the first look; now one more grace for the spill
    assert 0.9 <= took < 5.0, took


def test_waker_keeps_waiting_when_the_depositor_resolves_the_cycle(tmp_path, monkeypatch):
    import pytest

    from sglang.srt.managers.weg2_memory_saver import (
        VramCredit,
        Weg2XchgCreditCycleRefused,
    )

    monkeypatch.setenv(b1.ENV_CYCLE_GRACE_S, "3")
    credit = VramCredit("GPU-w109b-card2b", credit_dir=str(tmp_path / "credit"))
    t0 = time.monotonic()
    # the chain stands for 1 s, then the depositor has spilled and paused
    reader = lambda: list(_CHAIN) if time.monotonic() - t0 < 1.0 else None  # noqa: E731
    with pytest.raises(Exception) as exc:
        credit.wait_for(3202 * 1024 * 1024, budget_s=3.5, tag="weights_15",
                        free_bytes_now=0, cycle_reader=reader)
    # no W109: the wait ran on to its (short) budget -- in the boot the credit
    # of the spilled tag arrives instead
    assert not isinstance(exc.value, Weg2XchgCreditCycleRefused), exc.value
    assert time.monotonic() - t0 >= 3.0
