"""WAKE-RUNAHEAD-DEPOSITOR (barlink-lanes2-1023): the collector starts a tag
as soon as its depositor waits for it, not when an OLDER collect lets the wake
loop go on.

Metal (boot dkr27bnvfp4duorebarbar1duo10230932, ACS 0011, 300 BAR1 tag pairs,
49 under 10 GB/s): in the start-offset class the depositor's ring is full in a
few ms and it waits for free(0) -- its collector has not begun the tag. A
collector begins tag t only after the wake loop's main thread resumed t, and
that thread sat in the run-ahead bound (FIFO on the collect bound+1 places
back). Worst tag, P PP0 lane p1 ``8-weights_4``:

  D TP1 deposit weights_4 begins            1792748993.904 (ring full ~.91)
  PP0 main thread in the bound since          .808 (after submitting weights_3)
  ... waiting for weights_1's collect,       .772 -> .154 (held by lane c0)
  PP0 resume weights_4                       .162 -> .172, collect starts .177
  depositor lane-time: total 295 ms, wait 271 ms, rate 2.1 GB/s

Pinned here:
* the bound with ``early`` returns when the depositor flag appears (the
  repro of the offset above, scaled), the bound without it waits for the old
  collect -- the switch's two sides;
* never more than ONE collect beyond the bound; ``early`` false = FIFO; ANY
  keeps working; a failing collect still raises; a raising probe is ignored;
* the depositor posts ``ffwait.<seq>.0`` exactly while its first-free wait
  blocks, removes it after, says ``ffwait=N`` on its lane-time line, and posts
  nothing with the switch off;
* the wake loop's wiring (probe seq "<flip>-<next tag>", one extra worker,
  the WAKE-OVERLAP field).
"""
from __future__ import annotations

import concurrent.futures as cf
import ctypes
import os
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import bar1_lanes as b1  # noqa: E402
from flliper.srt.pdflip import wake_runahead as wr  # noqa: E402
from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[4]


def _futs(*states):
    out = []
    for i, s in enumerate(states):
        f = cf.Future()
        if s == "done":
            f.set_result(None)
        elif isinstance(s, BaseException):
            f.set_exception(s)
        out.append((f"weights_{i}", f))
    return out


def _later(f, s):
    t = threading.Timer(s, lambda: f.done() or f.set_result(None))
    t.daemon = True
    t.start()


def _settle(futs):
    for _t, f in futs:
        if not f.done():
            f.set_result(None)


class _Flag:
    """The depositor's ffwait flag as seen by the waker, up after ``at_s``."""

    def __init__(self, at_s=None):
        self.up = threading.Event()
        if at_s is not None:
            t = threading.Timer(at_s, self.up.set)
            t.daemon = True
            t.start()

    def __call__(self):
        return self.up.is_set()


# -- the switch ---------------------------------------------------------------

def test_switch_default_on():
    assert envs.FLLIPER_PDFLIP_WAKE_RUNAHEAD_DEPOSITOR.get() is True
    assert wr.runahead_depositor_on() is True
    assert b1.first_free_flag_on({}) is True
    assert b1.first_free_flag_on({b1.ENV_RUNAHEAD_DEPOSITOR: "0"}) is False
    assert b1.ENV_RUNAHEAD_DEPOSITOR == "FLLIPER_PDFLIP_WAKE_RUNAHEAD_DEPOSITOR"


# -- the repro of 8-weights_4, scaled 1:1 in seconds/1000 ---------------------

def _pp0_at_weights_3_bound():
    """PP0's futures after submitting weights_3 in flip 8: weights_0 and the
    no-op weights_7 done, weights_1/2/3 pending (their c0 lane waits for TP0)."""
    futs = _futs("done", "done", "pending", "pending", "pending")
    _later(futs[2][1], 0.354)   # weights_1 collect ends .154 (354 ms after .808)
    _later(futs[3][1], 0.530)   # weights_2 ends .338
    _later(futs[4][1], 0.587)   # weights_3 ends .395
    return futs


def test_repro_offset_old_bound_waits_for_the_older_collect():
    """Switch off (no ``early``): the main thread waits ~354 ms although the
    next tag's depositor has waited since ~100 ms -- the measured offset."""
    futs = _pp0_at_weights_3_bound()
    waited = wr.bound_wait(futs, 2, False)
    assert waited >= 0.33
    _settle(futs)


def test_repro_offset_depositor_releases_the_bound():
    """Switch on: the depositor of weights_4 waits from ~.904-.91 (96-102 ms
    after the bound began); the bound ends then, three collects in flight =
    bound + 1, so the early tag is the one allowed beyond the bound."""
    futs = _pp0_at_weights_3_bound()
    stats = {}
    t0 = time.perf_counter()
    waited = wr.bound_wait(futs, 2, False, early=_Flag(0.10), stats=stats)
    assert 0.09 <= waited < 0.2, waited
    assert time.perf_counter() - t0 < 0.25
    assert stats == {"early": 1}
    # the old collects are untouched -- only WHEN the loop goes on changed
    assert sum(1 for _t, f in futs if not f.done()) == 3
    _settle(futs)


def test_never_more_than_one_beyond_the_bound():
    """Four unfinished (one is already the early tag): the flag alone does not
    release; the first completion brings it to bound + 1 and then it does --
    before the FIFO target (weights_2 of the four) is done."""
    futs = _futs("pending", "pending", "pending", "pending")
    _later(futs[0][1], 0.15)
    _later(futs[1][1], 0.60)   # the FIFO target futs[-3]
    stats = {}
    waited = wr.bound_wait(futs, 2, False, early=lambda: True, stats=stats)
    assert 0.13 <= waited < 0.4, waited
    assert stats["early"] == 1
    _settle(futs)


def test_early_false_is_the_fifo_bound():
    futs = _futs("pending", "done", "done")
    _later(futs[0][1], 0.2)
    stats = {}
    waited = wr.bound_wait(futs, 2, False, early=lambda: False, stats=stats)
    assert waited >= 0.18 and stats == {}


def test_target_done_returns_without_asking():
    asked = []
    futs = _futs("done", "pending", "pending")
    assert wr.bound_wait(futs, 2, False, early=lambda: asked.append(1) or True) < 0.05
    assert asked == []
    _settle(futs)


def test_any_mode_with_early():
    futs = _futs("pending", "pending", "pending")
    _later(futs[2][1], 0.3)
    stats = {}
    waited = wr.bound_wait(futs, 2, True, early=_Flag(0.05), stats=stats)
    assert 0.04 <= waited < 0.2 and stats == {"early": 1}
    _settle(futs)


def test_failed_collect_still_raises():
    futs = _futs("pending", RuntimeError("W68 lane p1"), "pending")
    with pytest.raises(RuntimeError, match="W68"):
        wr.bound_wait(futs, 2, False, early=lambda: False)
    _settle(futs)


def test_a_raising_probe_is_ignored():
    futs = _futs("pending", "done", "done")
    _later(futs[0][1], 0.1)

    def boom():
        raise OSError("flag dir gone")

    waited = wr.bound_wait(futs, 2, False, early=boom)
    assert waited >= 0.08


# -- the depositor's flag (bar1_lanes) ----------------------------------------

PAIRS = ((0, 1), (1, 0))
SLOT = 4096


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


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


def _flat_descs(n, nbytes, tag="weights_4"):
    srcs = [bytearray(os.urandom(nbytes)) for _ in range(n)]
    dsts = [bytearray(nbytes) for _ in range(n)]
    descs = [SimpleNamespace(kind=tp.FLAT, nbytes=nbytes, src_off=0, dst_off=0,
                             param_name=f"w{i}", tag=tag,
                             src_ptr=_addr(srcs[i]), dst_ptr=_addr(dsts[i]))
             for i in range(n)]
    return descs, srcs, dsts


def _late_collector(tmp_path, ring, nbatches, collect_after_s, seq="8-weights_4"):
    """Depositor P rank 0 -> collector D rank 1 on lane p0; the collector
    begins ``collect_after_s`` late (its wake loop had not resumed the tag).
    A watcher records what the waker's probe answers meanwhile."""
    window = bytearray(SLOT * ring)
    dep = b1.Bar1Lanes("n1", "P", 0, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=_addr(window), slot_bytes=SLOT, ring=ring)
    dep.peers["p0"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=SLOT, ring=ring)
    descs, srcs, dsts = _flat_descs(nbatches, SLOT)
    out, dlog, seen = {}, [], {"waiting": False, "other_seq": False}
    stop = threading.Event()

    def _watch():
        while not stop.is_set():
            if col.depositor_waiting(seq):
                seen["waiting"] = True
            if col.depositor_waiting("8-weights_5"):
                seen["other_seq"] = True
            time.sleep(0.002)

    def _collect():
        time.sleep(collect_after_s)
        seen["waiting_at_collect_start"] = col.depositor_waiting(seq)
        out["c"] = b1.run_bar1_units(descs, _HostOps(), lanes=col, lane_key="p0", role="dst",
                                     seq=seq, phase="collect", budget_s=10.0,
                                     log=lambda *_a: None)

    w = threading.Thread(target=_watch, daemon=True)
    c = threading.Thread(target=_collect, daemon=True)
    w.start()
    c.start()
    out["d"] = b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p0", role="src",
                                 seq=seq, phase="deposit", budget_s=10.0, log=dlog.append)
    c.join(20)
    stop.set()
    w.join(5)
    assert not c.is_alive(), "collector hung"
    assert out == {"c": "", "d": ""}, out
    assert [bytes(x) for x in dsts] == [bytes(x) for x in srcs]
    flag = b1._flag_path(col.flags("p0", "dst"), b1.FFWAIT, seq, 0)
    return seen, dlog, flag


@pytest.mark.parametrize("ring,nbatches", [(3, 8), (4, 3)])
def test_depositor_announces_its_first_free_wait(tmp_path, monkeypatch, ring, nbatches):
    """(4, 3): a tag of <= ring batches -- the first free is the trailing one."""
    monkeypatch.delenv(b1.ENV_RUNAHEAD_DEPOSITOR, raising=False)
    seen, dlog, flag = _late_collector(tmp_path, ring, nbatches, 0.25)
    assert seen["waiting"] is True
    assert seen["waiting_at_collect_start"] is True
    assert seen["other_seq"] is False
    assert not os.path.exists(flag), "the flag must go when the wait ends"
    line = [x for x in dlog if "lane-time" in x][0]
    assert " ffwait=1" in line


def test_switch_off_posts_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_RUNAHEAD_DEPOSITOR, "0")
    seen, dlog, flag = _late_collector(tmp_path, 3, 8, 0.15)
    assert seen["waiting"] is False
    assert not os.path.exists(flag)
    line = [x for x in dlog if "lane-time" in x][0]
    assert " ffwait=off" in line


def test_no_flag_when_the_free_is_already_there(tmp_path, monkeypatch):
    monkeypatch.delenv(b1.ENV_RUNAHEAD_DEPOSITOR, raising=False)
    d = str(tmp_path / "flags")
    cred = b1.FileCredits(d, "8-weights_4")
    cred.send("free", 0)
    with b1.first_free_wait(cred, d, "8-weights_4", 0, "p0", 0) as posted:
        assert posted is False
    with b1.first_free_wait(b1.FileCredits(d, "8-weights_6"), d, "8-weights_6", 0, "p0", 0) as posted:
        assert posted is True
        assert os.path.exists(b1._flag_path(d, b1.FFWAIT, "8-weights_6", 0))
    assert not os.path.exists(b1._flag_path(d, b1.FFWAIT, "8-weights_6", 0))


def test_socket_credits_ready_reads_without_consuming():
    import socket as _s
    a, b = _s.socketpair()
    try:
        ca, cb = b1.SocketCredits(a), b1.SocketCredits(b)
        assert cb.ready("free", 0) is False
        ca.send("free", 0)
        time.sleep(0.01)
        assert cb.ready("free", 0) is True
        assert cb.recv("free", 0, 1.0) == {}
        assert cb.ready("free", 1) is False
    finally:
        a.close()
        b.close()


def test_depositor_waiting_reads_only_receiving_lanes(tmp_path):
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    assert col.depositor_waiting("3-weights_2") is False          # no receiving lane
    col.recv["p0"] = SimpleNamespace(dptr=0, slot_bytes=SLOT, ring=4)
    b1.post_flag(col.flags("p0", "dst"), b1.FFWAIT, "3-weights_2", 0, {})
    assert col.depositor_waiting("3-weights_2") is True
    assert col.depositor_waiting("3-weights_3") is False


# -- the wake loop's wiring ---------------------------------------------------

def test_wake_loop_wiring():
    src = (REPO / "python/flliper/srt/managers/scheduler_components/weight_updater.py").read_text()
    assert "_pdflip_runahead.runahead_depositor_on()" in src
    assert "early=_ra_early, stats=_ra_stats" in src
    import re as _re
    assert _re.search(r"self\._pdflip_runahead_depositor_probe\(\s*weights_tags\[_ti \+ 1\]\)", src)
    assert "except Exception:  # noqa: BLE001 -- no probe: the bound as before" in src
    assert "+ (1 if _ra_dep else 0)," in src     # the extra collect worker
    assert "runahead_depositor=%s" in src


def test_probe_names_the_next_tag_of_this_flip(tmp_path):
    from flliper.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as M,
    )
    col = b1.Bar1Lanes("n1", "D", 1, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))
    col.recv["p0"] = SimpleNamespace(dptr=0, slot_bytes=SLOT, ring=4)
    stub = SimpleNamespace(_pdflip_bar1=col, _pdflip_flip_index_now=8)
    probe = M._pdflip_runahead_depositor_probe(stub, "weights_4")
    assert probe() is False
    b1.post_flag(col.flags("p0", "dst"), b1.FFWAIT, "8-weights_4", 0, {})
    assert probe() is True
    # no flip index (desk counters) or no lanes: the bound as before
    assert M._pdflip_runahead_depositor_probe(
        SimpleNamespace(_pdflip_bar1=col, _pdflip_flip_index_now=None), "weights_4") is None
    assert M._pdflip_runahead_depositor_probe(
        SimpleNamespace(_pdflip_bar1=None, _pdflip_flip_index_now=8), "weights_4") is None
