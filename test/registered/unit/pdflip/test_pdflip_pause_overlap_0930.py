# SPDX-License-Identifier: Apache-2.0
"""PAUSE-SUB + PAUSE-OVERLAP (30.09., NF y4h/y4i): the sleeper's pause on the
3080 D ranks.

DER BEFUND: D's Schlaf-Leg bindet den D->P-Flip (y4h D-TP1 37/39, y4i D-TP2
27/36). ``pause_ms`` je Tag: 3080-D-Raenge 26-28 ms (Median, ~1 GiB), 5090
6 ms; Summe 465-517 ms je Leg. ``sync_ms`` (torch.cuda.synchronize VOR der
Pause, 52253301da) ist auf JEDEM Tag 0 -- das Geraet ist leer, die Zeit ist
die Unmap/Release-Arbeit selbst. P pausiert auf DERSELBEN Karte ein Drittel
je Byte; D's Expertenbaenke sind H95c-Spannen (ein Handle je Gitterzelle).

(1) PAUSE-SUB: der Saver zaehlt je Pause Allokationen, cuMemUnmap-Aufrufe und
    die Uhren von Unmap und Release (``tms_pause_stats``), die Zeile
    ``PDFLIP-PAUSE-SUB`` traegt sie -- gegen den Mock-Treiber gebaut.
(2) PAUSE-OVERLAP (Schalter aus): pause(t)+credit(t) auf einem Worker neben
    deposit(t+1), hoechstens eine Pause in Flug, Join vor jedem Deposit auf
    der On-Card-Lane und am Leg-Ende.
(3) das Was-waere-wenn im Leser ``dp_stage_legs``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import subprocess
import textwrap
import threading
import time
import types

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import pause_overlap as po
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

MIB = 1 << 20


# ---- (2) the overlap rule --------------------------------------------------------

def test_switch_default_on_after_metal_and_groups():
    # y4k + y4l metal: on by default; "off" stays an env value
    assert envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.get() is True
    assert envs.FLLIPER_PDFLIP_SLEEP_PAUSE_OVERLAP_GROUPS.get() == "D"
    assert po.overlap_on("D") is True
    with envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.override(False):
        assert po.overlap_on("D") is False
    with envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.override(True):
        assert po.overlap_on("D") is True
        assert po.overlap_on("P") is False
        with envs.FLLIPER_PDFLIP_SLEEP_PAUSE_OVERLAP_GROUPS.override("P,D"):
            assert po.overlap_on("P") is True


def test_one_pause_in_flight_in_tag_order_and_join_before_the_on_card_lane():
    events = []
    lock = threading.Lock()
    running = [0]
    peak = [0]

    def step(tag, ms):
        def _run():
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
                events.append(("pause-start", tag, time.perf_counter()))
            time.sleep(ms / 1000.0)
            with lock:
                events.append(("credit", tag, time.perf_counter()))
                running[0] -= 1
        return _run

    look = po.PauseOverlap({"w2"})
    # deposit(w0) happened; its pause goes to the worker
    look.submit("w0", step("w0", 60))
    assert look.before_deposit("w1") == 0.0   # no on-card lane: runs beside
    t_dep1 = time.perf_counter()
    look.submit("w1", step("w1", 30))           # joins w0 first ("order")
    waited = look.before_deposit("w2")          # on-card lane: joins w1
    t_dep2 = time.perf_counter()
    look.submit("w2", step("w2", 5))
    look.close()
    credits = [(e[1], e[2]) for e in events if e[0] == "credit"]
    starts = [(e[1], e[2]) for e in events if e[0] == "pause-start"]
    assert [c[0] for c in credits] == ["w0", "w1", "w2"]          # tag order kept
    assert peak[0] == 1                                            # never two pauses
    assert t_dep1 < dict(credits)["w0"]                            # deposit(w1) beside pause(w0)
    assert dict(credits)["w1"] <= t_dep2                           # w2's lane saw w1's credit
    assert dict(starts)["w1"] >= dict(credits)["w0"]
    assert waited > 0.0
    assert look.overlapped == 1 and look.submitted == 3
    assert [j[0] for j in look.joins] == ["order", "diag", "leg-end"]
    assert look.summary().startswith("PDFLIP-PAUSE-OVERLAP tags=3 overlapped=1")


def test_a_pause_that_raised_is_reraised_on_the_loop_thread():
    look = po.PauseOverlap(())

    def bad():
        raise RuntimeError("cuMemUnmap rc=1")

    look.submit("w0", bad)
    with pytest.raises(RuntimeError, match="cuMemUnmap"):
        look.submit("w1", lambda: None)   # the submit's join names it
    look.close()
    look2 = po.PauseOverlap(())
    look2.submit("w0", bad)
    with pytest.raises(RuntimeError):
        look2.close()


def test_chain_end_model():
    # (tag, rest, pause): the chain today = the sum
    steps = [("a", 80, 28), ("b", 80, 28), ("c", 10, 28), ("d", 80, 28)]
    end, dep = po.chain_end_ms(steps, {"c"}, 0.0, overlap=False)
    assert end == 108 + 108 + 38 + 108 and dep["d"] == 108 + 108 + 38 + 80
    # overlap, c on the on-card lane: a beside b, b joined before c,
    # c beside d, d's pause after d:  80 | 80 (a hidden) | join b @188 | c 198 |
    # submit c @198 | d 278, submit d: join c (226) -> pause d 278..306
    end, dep = po.chain_end_ms(steps, {"c"}, 0.0, overlap=True)
    assert dep == {"a": 80, "b": 160, "c": 198, "d": 278}
    assert end == 306
    # a pause longer than the next deposit spills onto the chain
    end, _ = po.chain_end_ms([("a", 10, 50), ("b", 10, 50)], (), 0.0, overlap=True)
    assert end == 10 + 50 + 50


# ---- (1) the saver's split, built against the mock driver --------------------------

def _h95c_mock():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_h95c_for_pause_sub", os.path.join(here, "test_pdflip_d_seat_vram_h95c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._MOCK


@pytest.fixture(scope="module")
def tms_lib(tmp_path_factory):
    gxx = shutil.which("g++")
    inc = "/usr/local/cuda/include"
    if gxx is None or not os.path.isfile(os.path.join(inc, "cuda.h")):
        pytest.skip("no g++ / CUDA headers to build the saver against a mock driver")
    import flliper.srt.pdflip as w

    src = os.path.join(os.path.dirname(w.__file__), "tms_csrc")
    out = tmp_path_factory.mktemp("tms_pause_sub")
    (out / "mock_cuda.cpp").write_text(textwrap.dedent(_h95c_mock()))
    so = out / "libtms_mock.so"
    cmd = [gxx, "-std=c++17", "-shared", "-fPIC", "-DUSE_CUDA", "-DTMS_HOOK_MODE_PRELOAD",
           "-I" + inc, "-I" + src] + [os.path.join(src, f) for f in (
               "core.cpp", "entrypoint.cpp", "host_ring.cpp", "api_forwarder.cpp")] + [
           str(out / "mock_cuda.cpp"), "-o", str(so), "-Wl,-Bsymbolic", "-ldl", "-lpthread"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    lib = ctypes.CDLL(str(so))
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    lib.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    lib.tms_pause.argtypes = [ctypes.c_char_p]
    lib.tms_resume_rc.argtypes = [ctypes.c_char_p]
    lib.tms_set_spans.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
                                  ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
    lib.mock_live_bytes.restype = ctypes.c_size_t
    return lib


def _malloc(lib, tag, size):
    lib.tms_set_current_tag(tag.encode())
    lib.tms_set_interesting_region(True)
    p = ctypes.c_void_p()
    assert lib.cudaMalloc(ctypes.byref(p), size) == 0
    lib.tms_set_interesting_region(False)
    return int(p.value)


def _stats(lib):
    fn = lib.tms_pause_stats
    fn.restype = ctypes.c_uint64
    buf = ctypes.create_string_buffer(64)
    a, u = ctypes.c_uint64(0), ctypes.c_uint64(0)
    um, rm, tm = ctypes.c_double(-1), ctypes.c_double(-1), ctypes.c_double(-1)
    seq = fn(buf, 64, ctypes.byref(a), ctypes.byref(u), ctypes.byref(um), ctypes.byref(rm), ctypes.byref(tm))
    return int(seq), buf.value.decode(), int(a.value), int(u.value), um.value, rm.value, tm.value


def test_pause_sub_counts_mappings_per_pause(tms_lib):
    lib = tms_lib
    seq0 = _stats(lib)[0]
    live0 = lib.mock_live_bytes()
    _malloc(lib, "ps_stock", 6 * MIB)
    _malloc(lib, "ps_stock", 4 * MIB)
    lib.tms_pause(b"ps_stock")
    seq, tag, allocs, unmaps, um, rm, tm = _stats(lib)
    assert seq == seq0 + 1 and tag == "ps_stock"
    assert (allocs, unmaps) == (2, 2)             # stock: one mapping per allocation
    assert um >= 0.0 and rm >= 0.0 and tm >= um + rm - 1e-6
    assert lib.mock_live_bytes() == live0           # the same release as before
    # a span-mapped allocation: one unmap per extent (the H95c lattice)
    p = _malloc(lib, "ps_span", 16 * MIB)
    lo = (ctypes.c_uint64 * 3)(0, 4 * MIB, 10 * MIB)
    hi = (ctypes.c_uint64 * 3)(2 * MIB, 8 * MIB, 14 * MIB)
    assert lib.tms_set_spans(ctypes.c_void_p(p), 3, lo, hi, 1) == 0
    lib.tms_pause(b"ps_span")
    seq, tag, allocs, unmaps, *_ = _stats(lib)
    assert (tag, allocs, unmaps) == ("ps_span", 1, 3)
    assert lib.mock_live_bytes() == live0
    assert lib.tms_resume_rc(b"ps_span") == 0      # the plan survives the pause
    assert lib.mock_live_bytes() - live0 == 10 * MIB
    lib.tms_pause(b"ps_span")
    assert _stats(lib)[3] == 3 and lib.mock_live_bytes() == live0


def test_adapter_reads_the_split_and_never_invents_one(tms_lib, monkeypatch):
    from flliper.srt.utils import torch_memory_saver_adapter as tma

    lib = tms_lib
    _malloc(lib, "ps_adapter", 2 * MIB)
    lib.tms_pause(b"ps_adapter")
    ad = object.__new__(tma._TorchMemorySaverAdapterReal)
    monkeypatch.setattr(tma, "_pdflip_ring_symbol", lambda name: getattr(lib, name, None))
    st = ad.pause_stats("ps_adapter")
    assert st["allocations"] == 1 and st["unmaps"] == 1 and st["total_ms"] >= 0.0
    assert ad.pause_stats("another_tag") is None       # the record is not this tag's
    monkeypatch.setattr(tma, "_pdflip_ring_symbol", lambda name: None)
    assert ad.pause_stats("ps_adapter") is None          # stock hook: absence
    assert tma._TorchMemorySaverAdapterNoop().pause_stats("x") is None


# ---- the wiring in the sleep leg ---------------------------------------------------

def _wu():
    from flliper.srt.managers.scheduler_components import weight_updater as wu
    return wu


class _Stub:
    def __init__(self, diag, group="D"):
        self._diag = diag
        self._group = group
        self.census_calls = []

    def _pdflip_group_name(self):
        return self._group

    def _pdflip_xchg_diag_tags(self, flip_index):
        return self._diag

    def _pdflip_device_index(self):
        return -1

    def _pdflip_tag_resident_bytes(self, tag):
        self.census_calls.append(tag)
        return 7


def _scope(stub, h111b=None):
    wu = _wu()
    req = types.SimpleNamespace(epoch="1790755908.3")
    return wu.SchedulerWeightUpdaterManager._pdflip_pause_overlap_scope(stub, req, ["w0", "w1"], h111b)


def test_scope_is_the_chain_unless_armed_readable_and_not_h111b():
    stub = _Stub({"w1"})
    with envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.override(False):
        with _scope(stub) as look:
            assert look is None                       # switch off
    with envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.override(True):
        with _scope(stub, h111b=object()) as look:
            assert look is None                       # H111b runs the leg
        with _scope(_Stub(None)) as look:
            assert look is None                       # lanes unreadable
        with _scope(_Stub({"w1"}, group="P")) as look:
            assert look is None                       # group not listed
        with _scope(stub) as look:
            assert isinstance(look, po.PauseOverlap) and look.diag_tags == {"w1"}
            assert stub._pdflip_resident_prefetch == {"w0": 7, "w1": 7}
            look.submit("w0", lambda: None)
        assert stub._pdflip_resident_prefetch is None
        assert look.joins[-1][0] == "leg-end"


def test_scope_keeps_the_loops_own_exception():
    stub = _Stub(set())

    def bad():
        raise RuntimeError("pause fault")

    with envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.override(True):
        with pytest.raises(ValueError, match="W106"):
            with _scope(stub) as look:
                look.submit("w0", bad)
                raise ValueError("W106 deposit refused")
    assert stub._pdflip_resident_prefetch is None


def test_resident_census_reads_the_prefetch_while_armed():
    wu = _wu()
    calls = []
    stub = types.SimpleNamespace(
        memory_saver_adapter=types.SimpleNamespace(tag_bytes=lambda t: calls.append(t) or 5),
        _pdflip_resident_prefetch={"w0": 0})
    f = wu.SchedulerWeightUpdaterManager._pdflip_tag_resident_bytes
    assert f(stub, "w0") == 0 and calls == []            # no saver call (its mutex)
    assert f(stub, "w9") == 5 and calls == ["w9"]
    stub._pdflip_resident_prefetch = None
    assert f(stub, "w0") == 5


def test_the_sleep_loop_is_wired():
    wu = _wu()
    src = open(wu.__file__).read()
    i = src.index("self._pdflip_h111b_scope(recv_req, weights_tags) as _h111b, \\")
    body = src[i:i + 12000]
    assert "self._pdflip_pause_overlap_scope(recv_req, weights_tags, _h111b) as _po:" in body
    # the join before the deposit, the pause+credit order inside the step
    assert body.index("_po.before_deposit(tag)") < body.index("_t_dep0 = time.perf_counter()")
    step = body[body.index("def _pdflip_pause_step("):body.index("_tag_stall.disarm(_stall)")]
    assert step.index("self.memory_saver_adapter.pause(tag)") < step.index("credit.publish(tag")
    assert "self._pdflip_pause_sub_line(tag, pdflip_per_tag[tag][1])" in step
    assert "if _po is None:\n                        _t_prev_end = _pdflip_pause_step()" in step
    assert "_po.submit(tag, _pdflip_pause_step)" in step
    # the sync stays on the loop thread, before the step is handed over
    assert body.index("torch.cuda.synchronize()") < body.index("def _pdflip_pause_step(")


def test_pause_sub_line(caplog):
    wu = _wu()
    import logging

    stub = types.SimpleNamespace(memory_saver_adapter=types.SimpleNamespace(
        pause_stats=lambda t: {"allocations": 3, "unmaps": 41, "unmap_ms": 19.5,
                               "release_ms": 5.25, "total_ms": 25.0}))
    with caplog.at_level(logging.INFO):
        wu.SchedulerWeightUpdaterManager._pdflip_pause_sub_line(stub, "weights_3", 26.0)
    assert "PDFLIP-PAUSE-SUB tag=weights_3 allocs=3 unmaps=41 unmap_ms=19.5 release_ms=5.2" in caplog.text
    caplog.clear()
    stub.memory_saver_adapter = types.SimpleNamespace(pause_stats=lambda t: None)
    wu.SchedulerWeightUpdaterManager._pdflip_pause_sub_line(stub, "weights_3", 26.0)
    assert "PDFLIP-PAUSE-SUB" not in caplog.text


# ---- (3) the reader's what-if --------------------------------------------------------

def test_reader_what_if_pause_overlap():
    from flliper.srt.pdflip.tools import dp_stage_legs as dsl

    T0 = 1000.0
    order = ("a", "b", "c")

    def recs(costs, pauses):
        out, t = [], T0
        for tag, c, pz in zip(order, costs, pauses):
            out.append({"tag": tag, "deposit_ms": c - pz, "pause_ms": pz, "total_ms": c,
                        "t0": t, "t": t + c / 1000.0})
            t += c / 1000.0
        return out

    flip = {
        "begin": T0 - 0.2, "done": T0 + 1.0,
        "p": {"PP0": [{"tag": t, "t0": T0, "t": T0 + 0.01} for t in order],
              "PP1": [{"tag": t, "t0": T0, "t": T0 + 0.01} for t in order]},
        "bytes": {"PP0": {"a": 100, "c": 100}, "PP1": {"b": 100}},
        "d": {"TP0": recs((20, 20, 20), (5, 5, 5)),
              "TP1": recs((100, 100, 100), (30, 30, 30))},
    }
    w = dsl.what_if_pause_overlap(flip)
    assert w["now"] == pytest.approx({"TP0": 60, "TP1": 300})
    # TP1's on-card waker is PP1 (tag b): a hidden beside b? no -- b is on
    # the lane: join a (70+30=100) -> b 170, submit; c 240 (b hidden),
    # submit c: join b (200) -> c's pause 240..270
    assert w["overlap"]["TP1"] == pytest.approx(270)
    assert w["crit_now"] == "TP1" and w["legs_end_now"] - w["legs_end_overlap"] == pytest.approx(30)
