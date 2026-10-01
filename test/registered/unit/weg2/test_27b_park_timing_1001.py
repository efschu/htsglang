"""PARK-TIMING (01.10.): the D park RPC (``#1476 DISPATCH
kind=Weg2ParkRunningReqInput``) sat 0.57-1.84 s ahead of every D->P flip
(z30y11/y12/y13, equal on all ranks), and its log had one-second stamps only.
``park_running`` now prints ONE ``WEG2-D-PARK TIMING`` line per park per rank
with monotonic sub-times: land / draft / retract (forced host write-through)
/ note / #248 mark chain (chain walk + mark write) / #59b probe / #59b tp-min
collective / rest, plus seq tokens, resumable pages and the host slots/bytes
the write-through took.

Driven through the production path: the real ``park_running``, the real
``park_l3.mark_parked`` (TP0 writer), the real ``park_depths`` -> ``group_depths``
in tp-min mode; only the leaves (the retraction, the tree walk, the mark
file, the admission probe, the gloo reduce) are stand-ins that advance one
fake monotonic clock by a known amount.
"""
from __future__ import annotations

import logging
import os
import re
import time
import types
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import tp_match_floor  # noqa: E402
from sglang.srt.managers import weg2_resumable_depth as wrd  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import handoff_pending as hp  # noqa: E402
from sglang.srt.weg2 import park_l3  # noqa: E402

RETRACT_S, CHAIN_S, WRITE_S, PROBE_S, COLL_S = 0.5, 0.06, 0.04, 0.03, 0.25
HOST_TAKEN, BYTES_PER_SLOT = 1000, 4096


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def adv(self, s):
        self.t += s


@pytest.fixture
def clk(monkeypatch):
    c = _Clock()
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.setattr(rt, "_park_clock", c)
    monkeypatch.setattr(park_l3, "time", types.SimpleNamespace(monotonic=c, time=time.time))
    monkeypatch.setattr(park_l3, "enabled", lambda: True)

    def _mark_park(rid, chain, page_size=0, page_range=None):
        c.adv(WRITE_S)
        return True

    monkeypatch.setattr(hp, "mark_park", _mark_park)

    def _probe(tree, view, follow=False):
        c.adv(PROBE_S)
        return len(view.origin_input_ids) + len(view.output_ids) - 2

    monkeypatch.setattr(tp_match_floor, "admission_probe", _probe)
    monkeypatch.setattr(tp_match_floor, "form_a_follow_active", lambda: False)

    def _tp_min(values):
        c.adv(COLL_S)
        return [int(v) - 1 for v in values]  # the group's MIN: one lower than this rank

    monkeypatch.setattr(wrd, "_tp_min", _tp_min)
    return c


class _HostPool:
    def __init__(self):
        self.free, self.size_per_token = 50_000, BYTES_PER_SLOT

    def available_size(self):
        return self.free


class _Tree:
    page_size = 1
    is_eagle = False

    def __init__(self, clk):
        self.clk = clk
        self.root_node = types.SimpleNamespace(parent=None)
        self.cache_controller = types.SimpleNamespace(mem_pool_host=_HostPool())

    def match_prefix(self, params):
        self.clk.adv(CHAIN_S / 2)
        n = len(params.key)
        node = types.SimpleNamespace(parent=self.root_node, hash_value=[f"k{i}" for i in range(n)])
        return types.SimpleNamespace(last_host_node=None, last_device_node=node)


class _Batch:
    def __init__(self, reqs, tree):
        self.reqs, self.batch_is_full, self.tree = list(reqs), True, tree

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        assert retain and not offload_kv
        self.tree.clk.adv(RETRACT_S)
        if self.tree.cache_controller is not None:
            self.tree.cache_controller.mem_pool_host.free -= HOST_TAKEN
        out, self.reqs = self.reqs, []
        return out


class _Sched:
    def __init__(self, running, clk, rank):
        self.tree_cache = _Tree(clk)
        self.running_batch = _Batch(running, self.tree_cache)
        self.waiting_queue, self.last_batch, self.enable_overlap = [], None, False
        self.result_queue, self.chunked_req, self.anchor_tails = deque(), None, []
        self.server_args, self.weg2_dormant = types.SimpleNamespace(), False
        self.page_size, self.tp_rank = 1, rank
        self.ps = types.SimpleNamespace(pp_size=1, attn_dp_size=1, tp_size=3, tp_rank=rank,
                                        attn_tp_rank=rank)

    def _969ad_note_retract(self, req, site):
        pass


def _req(rid, n_in, n_out):
    return types.SimpleNamespace(rid=rid, kv_arrival_seq=1, origin_input_ids=[1] * n_in,
                                 output_ids=[2] * n_out, is_fast_lane=False, spill_class=None)


def _timing_lines(caplog):
    return [r.getMessage() for r in caplog.records if "WEG2-D-PARK TIMING" in r.getMessage()]


def _fields(line):
    return {k: v for k, v in re.findall(r"(\w+)=(\S+)", line)}


def _park(clk, caplog, rank):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    reqs = [_req("weg2-10-26", 30000, 344), _req("weg2-16-44", 55000, 255)]
    s = _Sched(reqs, clk, rank)
    with caplog.at_level(logging.INFO):
        out = rt.park_running(s, Weg2ParkRunningReqInput(epoch=18, reason="immediate-over-x"))
    assert out.success and set(out.parked) == {"weg2-10-26", "weg2-16-44"}
    return out, _timing_lines(caplog)


def test_tp0_park_names_every_step_with_monotonic_sub_times(clk, caplog):
    out, lines = _park(clk, caplog, rank=0)
    assert len(lines) == 1, "exactly one TIMING line per park per rank"
    f = _fields(lines[0])
    assert f["epoch"] == "18" and f["rank"] == "0"
    assert float(f["retract_ms"]) == pytest.approx(RETRACT_S * 1e3)
    # TP0 writes the #248 marks: chain walk and mark write, per request
    assert float(f["mark_chain_ms"]) == pytest.approx(2 * CHAIN_S / 2 * 1e3)
    assert float(f["mark_write_ms"]) == pytest.approx(2 * WRITE_S * 1e3)
    assert float(f["mark_ms"]) == pytest.approx((CHAIN_S + 2 * WRITE_S) * 1e3)
    assert int(f["mark_keys"]) == 30344 + 55255
    # #59b: two local probes, ONE timed tp-min reduce (the default reduce's result kept)
    assert float(f["probe_ms"]) == pytest.approx(2 * PROBE_S * 1e3)
    assert float(f["coll_ms"]) == pytest.approx(COLL_S * 1e3)
    assert out.weg2_resumable_depth == {"weg2-10-26": 30344 - 3, "weg2-16-44": 55255 - 3}
    # the laps are contiguous: their sum is the total
    parts = ("land_ms", "draft_ms", "retract_ms", "note_ms", "mark_ms", "probe_ms", "coll_ms", "rest_ms")
    assert float(f["total_ms"]) == pytest.approx(sum(float(f[k]) for k in parts), abs=0.5)
    assert float(f["total_ms"]) == pytest.approx(
        (RETRACT_S + CHAIN_S + 2 * WRITE_S + 2 * PROBE_S + COLL_S) * 1e3, abs=0.5)
    # sizes: seq tokens, resumable pages, what the write-through took on the host
    assert int(f["seq_tokens"]) == 30344 + 55255
    assert int(f["resumable_tokens"]) == (30344 - 3) + (55255 - 3) == int(f["pages"])
    assert f["wt_host_slots"] == str(HOST_TAKEN)
    assert f["wt_bytes"] == str(HOST_TAKEN * BYTES_PER_SLOT)
    assert f["running"] == "2" and f["retracted"] == "2"


def test_every_rank_prints_the_same_fields_tp1_waits_in_the_collective(clk, caplog):
    _, l0 = _park(clk, caplog, rank=0)
    caplog.clear()
    _, l1 = _park(clk, caplog, rank=1)
    assert len(l1) == 1
    f0, f1 = _fields(l0[0]), _fields(l1[0])
    assert list(f0) == list(f1), "rank-uniform: the same fields in the same order"
    assert f1["rank"] == "1"
    # TP1 writes no marks -- its chain cost is 0, its wait shows up in coll_ms
    assert float(f1["mark_ms"]) == 0.0 and f1["mark_keys"] == "0"
    assert float(f1["coll_ms"]) == pytest.approx(COLL_S * 1e3)


def test_no_host_tier_prints_na_and_parks_anyway(clk, caplog):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    s = _Sched([_req("weg2-1-1", 100, 3)], clk, 0)
    s.tree_cache.cache_controller = None
    with caplog.at_level(logging.INFO):
        out = rt.park_running(s, Weg2ParkRunningReqInput(epoch=3, reason="x"))
    assert out.success
    f = _fields(_timing_lines(caplog)[0])
    assert f["wt_host_slots"] == "na" and f["wt_bytes"] == "na"
