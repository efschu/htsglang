"""BAR1 lanes (18.09.2026): the pure decisions, the flag credits, the per-tag
mode agreement and the ring transport end to end on host memory (fake ops)."""
from __future__ import annotations

import ctypes
import os
import threading
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def test_knobs_and_sizing():
    assert b1.lanes_on({}) and not b1.lanes_on({b1.ENV_ON: "0"})
    assert b1.ring_slots({}) == 4 and b1.ring_slots({b1.ENV_RING: "99"}) == 16
    assert b1.ring_slots({b1.ENV_RING: "x"}) == 4 and b1.ring_slots({b1.ENV_RING: "1"}) == 3
    assert b1.slot_bytes_for(256 << 20, {}) == 4 << 20          # a 3080's BAR1: 4 x 4 = the one 16-MiB hold
    assert b1.slot_bytes_for(32 << 30, {}) == 32 << 20          # the 5090's
    assert b1.slot_bytes_for(256 << 20, {b1.ENV_SMALL_SLOT_MIB: "16"}) == 16 << 20


def test_small_bar_lanes_borrow_the_group_windows():
    big = [0]
    d_win = {"world:0": 16 << 20, "tp:0": 32 << 20, "dcp:0": 40 << 20}
    assert b1.borrow_plan("D", "p0", PAIRS, big, d_win) == ("tp:0", 0, 32 << 20, 8 << 20)
    assert b1.borrow_plan("D", "p5", PAIRS, big, d_win) == ("dcp:0", 0, 40 << 20, 10 << 20)
    assert b1.borrow_plan("D", "p5", PAIRS, big, {"tp:0": 32 << 20}) == ("tp:0", 0, 32 << 20, 8 << 20)
    assert "no tp/dcp" in b1.borrow_plan("D", "p0", PAIRS, big, {"world:0": 16 << 20})
    p_win = {"world:0": 24 << 20, "pp:0": 96 << 20}
    # PP1 (card 1) receives p0 (from the 5090) and p5 (from card 2): the two halves of pp
    assert b1.borrow_plan("P", "p0", PAIRS, big, p_win) == ("pp:0", 0, 48 << 20, 12 << 20)
    assert b1.borrow_plan("P", "p5", PAIRS, big, p_win) == ("pp:0", 48 << 20, 48 << 20, 12 << 20)
    assert "no pp" in b1.borrow_plan("P", "p0", PAIRS, big, {"world:0": 24 << 20})
    assert "too small" in b1.borrow_plan("D", "p0", PAIRS, big, {"tp:0": 4 << 20})
    assert "not a cross lane" in b1.borrow_plan("D", "c1", PAIRS, big, d_win)


def test_roles_follow_the_directed_pair_and_only_cross_lanes():
    assert b1.lane_role("p0", 0, PAIRS) == "src"
    assert b1.lane_role("p0", 1, PAIRS) == "dst"
    assert b1.lane_role("p0", 2, PAIRS) is None
    assert b1.lane_role("c1", 1, PAIRS) is None
    assert b1.lane_role("p9", 1, PAIRS) is None
    assert b1.ring_slot(5, 2) == 1 and b1.ring_slot(4, 2) == 0


def test_lane_dir_is_keyed_by_the_receiving_group(tmp_path):
    p_side = b1.lane_dir("n1", "p0", "P", str(tmp_path))
    d_side = b1.lane_dir("n1", "p0", "D", str(tmp_path))
    assert p_side != d_side
    assert b1.socket_path("n1", "p0", "D", str(tmp_path)).startswith(d_side)
    assert b1.other_group("P") == "D" and b1.other_group("D") == "P"


def test_flags_are_consumed_once_and_carry_a_payload(tmp_path):
    d = str(tmp_path / "flags")
    assert b1.take_flag(d, "full", 3, 0, 0.05) is None
    b1.post_flag(d, "full", 3, 0, {"rec": [1, 2]})
    assert b1.take_flag(d, "full", 3, 0, 1.0) == {"rec": [1, 2]}
    assert b1.take_flag(d, "full", 3, 0, 0.05) is None          # consumed
    b1.post_flag(d, "free", 3, 1)
    assert b1.take_flag(d, "free", 3, 1, 1.0) == {}
    assert b1.take_flag(d, "free", 3, 1, 0.5, liveness=lambda: False) is None


def _lanes(tmp_path, group, rank):
    return b1.Bar1Lanes("n1", group, rank, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))


def test_mode_agreement_depositor_decides_collector_follows(tmp_path):
    dep = _lanes(tmp_path, "P", 0)       # PP0 deposits on p0 into TP1
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = SimpleNamespace(dptr=1, slot_bytes=8, ring=2)
    # without a mapped peer the depositor says host, the collector reads it
    assert dep.lane_mode("p0", "src", seq=0) == b1.MODE_HOST
    assert col.lane_mode("p0", "dst", seq=0, timeout_s=1.0) == b1.MODE_HOST
    dep.peers["p0"] = SimpleNamespace(dev_ptr=1, slot_bytes=8, ring=2)
    assert dep.lane_mode("p0", "src", seq=1) == b1.MODE_BAR1
    assert col.lane_mode("p0", "dst", seq=1, timeout_s=1.0) == b1.MODE_BAR1
    # a collector that sees no mode file within its budget runs the host path
    assert col.lane_mode("p0", "dst", seq=2, timeout_s=0.05) == b1.MODE_HOST
    # the mode file is keyed by the RECEIVING group: PP1 (P, dst of p0) does
    # not read TP1's file
    dep.lane_mode("p0", "src", seq=3)
    assert _lanes(tmp_path, "P", 1).lane_mode("p0", "dst", seq=3, timeout_s=0.05) == b1.MODE_HOST
    assert col.lane_mode("p0", "dst", seq=3, timeout_s=1.0) == b1.MODE_BAR1


class _HostOps:
    """memcpy on host addresses -- the ring transport's copies, byte for byte."""

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


@pytest.mark.parametrize("ring", [2, 3, 4, 8])
def test_ring_transport_moves_every_byte_flat_and_strided(tmp_path, ring):
    slot = 4096
    window = bytearray(slot * ring)
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1)
    # both ends see the same window (host memory stands in for the BAR)
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=slot, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=slot, ring=ring)
    # a FLAT tensor of 3.2 slots and a STRIDED2D one (rows of 1000 B in a
    # 1536-B pitch, scattered into a 2048-B pitch), plus a no-write unit
    flat_src = bytearray(os.urandom(13000))
    flat_dst = bytearray(13000)
    rows, run, spitch, dpitch = 30, 1000, 1536, 2048
    s2_src = bytearray(os.urandom(rows * spitch))
    s2_dst = bytearray(rows * dpitch)
    nw_src = bytearray(os.urandom(500))
    nw_dst = bytearray(500)
    descs = [
        SimpleNamespace(kind=tp.FLAT, nbytes=13000, src_off=0, dst_off=0, param_name="w.flat",
                        tag="t0", src_ptr=_addr(flat_src), dst_ptr=_addr(flat_dst)),
        SimpleNamespace(kind=tp.STRIDED2D, nbytes=rows * run, rows=rows, run_bytes=run,
                        spitch=spitch, dpitch=dpitch, src_off=0, dst_off=0, param_name="w.s2",
                        tag="t0", src_ptr=_addr(s2_src), dst_ptr=_addr(s2_dst)),
        SimpleNamespace(kind=tp.FLAT, nbytes=500, src_off=0, dst_off=0, param_name="w.nw",
                        tag="t0", src_ptr=_addr(nw_src), dst_ptr=_addr(nw_dst)),
    ]
    ops = _HostOps()
    out = {}

    def _collect():
        out["c"] = b1.run_bar1_units(descs, ops, lanes=col, lane_key="p0", role="dst", seq=7,
                                     phase="collect", no_write={("t0", "w.nw")}, budget_s=5.0,
                                     log=lambda *_a: None)
    th = threading.Thread(target=_collect)
    th.start()
    out["d"] = b1.run_bar1_units(descs, ops, lanes=dep, lane_key="p0", role="src", seq=7,
                                 phase="deposit", budget_s=5.0, log=lambda *_a: None)
    th.join(10)
    assert out["d"] == "" and out["c"] == ""
    assert bytes(flat_dst) == bytes(flat_src)
    for r in range(rows):
        assert s2_dst[r * dpitch:r * dpitch + run] == s2_src[r * spitch:r * spitch + run]
        assert s2_dst[r * dpitch + run:(r + 1) * dpitch] == bytes(dpitch - run)  # only payload
    assert bytes(nw_dst) == bytes(500)                       # consumed, not written
    # nothing left behind: every flag consumed, the ring's tail freed
    d = dep.flags("p0", "src")
    assert not [f for f in os.listdir(d) if not f.endswith(".tmp")]


def test_collector_refuses_a_disagreeing_plan(tmp_path):
    slot, ring = 4096, 2
    window = bytearray(slot * ring)
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=slot, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=slot, ring=ring)
    src = bytearray(os.urandom(3000))
    dst = bytearray(3000)
    d_descs = [SimpleNamespace(kind=tp.FLAT, nbytes=3000, src_off=0, dst_off=0, param_name="a",
                               tag="t", src_ptr=_addr(src), dst_ptr=None)]
    c_descs = [SimpleNamespace(kind=tp.FLAT, nbytes=3000, src_off=0, dst_off=0, param_name="b",
                               tag="t", src_ptr=None, dst_ptr=_addr(dst))]
    ops = _HostOps()
    assert b1.run_bar1_units(d_descs, ops, lanes=dep, lane_key="p0", role="src", seq=1,
                             phase="deposit", budget_s=2.0, log=lambda *_a: None) == ""
    why = b1.run_bar1_units(c_descs, ops, lanes=col, lane_key="p0", role="dst", seq=1,
                            phase="collect", budget_s=2.0, log=lambda *_a: None)
    assert "identity mismatch" in why and bytes(dst) == bytes(3000)


def test_no_window_on_this_side_is_a_named_refusal(tmp_path):
    dep = _lanes(tmp_path, "P", 0)
    why = b1.run_bar1_units([], _HostOps(), lanes=dep, lane_key="p0", role="src", seq=0,
                            phase="deposit", log=lambda *_a: None)
    assert "no window" in why
