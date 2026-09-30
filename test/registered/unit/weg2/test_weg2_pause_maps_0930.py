# SPDX-License-Identifier: Apache-2.0
"""PAUSE-MAPS (30.09., NF y4i, tms_csrc patch 5): the sleeper's pause releases
an H95c span-mapped allocation with ONE cuMemUnmap per contiguous run of
extents instead of one per extent.

DER BEFUND (WEG2-PAUSE-SUB, y4i 09301011, 12 D->P-Flips): ein 3080-D-Tag hat
10-12 Allokationen; vor der ersten D-Phase (ein Extent je Allokation) 10
cuMemUnmap-Aufrufe, unmap_ms 10.6; danach (H95c-Gitter) 31-64 Aufrufe,
21-25 ms. Die Kosten folgen den Treiberaufrufen, nicht den Bytes (P 3.1 GiB
je Tag ~1.1 ms je Aufruf wie D 0.8 GiB). Das Gitter bleibt (ein Live-Schrumpf
behaelt nur ganze Extents), die Zahl der Unmap-Aufrufe nicht.

(1) der Saver gegen einen Mock-Treiber, der einen Bereich ueber mehrere
    aneinanderliegende Mappings in EINEM Aufruf entmappt (wie der echte,
    CUDA-Sample multi-device mmap) -- oder ihn verweigert (Rueckfall);
(2) der Schalter (Default aus) und sein Weg in den Saver;
(3) der Adapter und die Zeile.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import shutil
import subprocess
import textwrap
import types

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

MIB = 1 << 20

_UNMAP_STOCK = r'''CUresult cuMemUnmap(CUdeviceptr a, size_t size) {
  auto it = g_map.find(a); if (it == g_map.end() || it->second.first != size) return CUDA_ERROR_INVALID_VALUE;
  g_map.erase(it); return CUDA_SUCCESS; }'''

_UNMAP_MULTI = r'''static int g_multi = 1; static long g_unmap_calls = 0;
void mock_set_unmap_multi(int on) { g_multi = on; }
long mock_unmap_calls() { return g_unmap_calls; }
CUresult cuMemUnmap(CUdeviceptr a, size_t size) {
  ++g_unmap_calls;
  auto it = g_map.find(a); if (it == g_map.end()) return CUDA_ERROR_INVALID_VALUE;
  if (it->second.first == size) { g_map.erase(it); return CUDA_SUCCESS; }
  if (!g_multi) return CUDA_ERROR_INVALID_VALUE;
  std::vector<CUdeviceptr> keys; CUdeviceptr cur = a;
  while (cur < a + size) {
    auto m = g_map.find(cur); if (m == g_map.end()) return CUDA_ERROR_INVALID_VALUE;
    keys.push_back(cur); cur += m->second.first; }
  if (cur != a + size) return CUDA_ERROR_INVALID_VALUE;
  for (auto k : keys) g_map.erase(k);
  return CUDA_SUCCESS; }'''


def _mock_source() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_h95c_for_pause_maps", os.path.join(here, "test_weg2_d_seat_vram_h95c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = textwrap.dedent(mod._MOCK)
    assert _UNMAP_STOCK in src, "the H95c mock's cuMemUnmap changed -- update this test"
    return src.replace("#include <map>", "#include <map>\n#include <vector>").replace(
        _UNMAP_STOCK, _UNMAP_MULTI)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    gxx = shutil.which("g++")
    inc = "/usr/local/cuda/include"
    if gxx is None or not os.path.isfile(os.path.join(inc, "cuda.h")):
        pytest.skip("no g++ / CUDA headers to build the saver against a mock driver")
    import sglang.srt.weg2 as w

    src = os.path.join(os.path.dirname(w.__file__), "tms_csrc")
    out = tmp_path_factory.mktemp("tms_pause_maps")
    (out / "mock_cuda.cpp").write_text(_mock_source())
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
    lib.mock_mapped_bytes.restype = ctypes.c_size_t
    lib.mock_unmap_calls.restype = ctypes.c_long
    lib.mock_set_unmap_multi.argtypes = [ctypes.c_int]
    return lib


def _malloc(lib, tag, size):
    lib.tms_set_current_tag(tag.encode())
    lib.tms_set_interesting_region(True)
    p = ctypes.c_void_p()
    assert lib.cudaMalloc(ctypes.byref(p), size) == 0
    lib.tms_set_interesting_region(False)
    return int(p.value)


def _lattice(lib, tag):
    """One 16 MiB bank allocation with the H95c lattice: three back-to-back
    cells [0,4) [4,6) [6,8) MiB and one apart [10,14) -- four extents, two runs."""
    p = _malloc(lib, tag, 16 * MIB)
    lo = (ctypes.c_uint64 * 4)(0, 4 * MIB, 6 * MIB, 10 * MIB)
    hi = (ctypes.c_uint64 * 4)(4 * MIB, 6 * MIB, 8 * MIB, 14 * MIB)
    assert lib.tms_set_spans(ctypes.c_void_p(p), 4, lo, hi, 1) == 0
    return p


def _sub(lib):
    fn = lib.tms_pause_stats
    fn.restype = ctypes.c_uint64
    buf = ctypes.create_string_buffer(64)
    a, u = ctypes.c_uint64(0), ctypes.c_uint64(0)
    um, rm, tm = ctypes.c_double(0), ctypes.c_double(0), ctypes.c_double(0)
    seq = fn(buf, 64, ctypes.byref(a), ctypes.byref(u), ctypes.byref(um), ctypes.byref(rm), ctypes.byref(tm))
    return int(seq), buf.value.decode(), int(a.value), int(u.value)


def _maps(lib):
    fn = lib.tms_pause_maps_stats
    fn.restype = ctypes.c_uint64
    buf = ctypes.create_string_buffer(64)
    e, r, f = ctypes.c_uint64(0), ctypes.c_uint64(0), ctypes.c_uint64(0)
    c = ctypes.c_int(-1)
    seq = fn(buf, 64, ctypes.byref(e), ctypes.byref(r), ctypes.byref(f), ctypes.byref(c))
    return int(seq), buf.value.decode(), int(e.value), int(r.value), int(f.value), int(c.value)


def _pause(lib, tag):
    c0 = lib.mock_unmap_calls()
    lib.tms_pause(tag.encode())
    return lib.mock_unmap_calls() - c0


# ---- (1) the saver against the mock driver -------------------------------------------

def test_off_is_the_patch4_walk_call_for_call(lib):
    lib.tms_set_pause_coalesce(0)
    live0 = lib.mock_live_bytes()
    _lattice(lib, "pm_off")
    assert lib.mock_live_bytes() - live0 == 12 * MIB
    assert _pause(lib, "pm_off") == 4                   # one call per extent
    seq, tag, allocs, unmaps = _sub(lib)
    assert (tag, allocs, unmaps) == ("pm_off", 1, 4)
    mseq, mtag, ext, runs, fb, co = _maps(lib)
    assert (mseq, mtag, ext, runs, fb, co) == (seq, "pm_off", 4, 0, 0, 0)
    assert lib.mock_live_bytes() == live0


def test_on_one_unmap_per_run_same_release(lib):
    assert lib.tms_set_pause_coalesce(1) == 1
    try:
        live0 = lib.mock_live_bytes()
        _lattice(lib, "pm_on")
        assert _pause(lib, "pm_on") == 2                # [0,8) in one call, [10,14) alone
        seq, tag, allocs, unmaps = _sub(lib)
        assert (tag, allocs, unmaps) == ("pm_on", 1, 2)
        assert _maps(lib)[1:] == ("pm_on", 4, 1, 0, 1)
        assert lib.mock_live_bytes() == live0            # every handle released
        # the plan survives: the next resume maps the same four cells again
        assert lib.tms_resume_rc(b"pm_on") == 0
        assert lib.mock_live_bytes() - live0 == 12 * MIB
        assert _pause(lib, "pm_on") == 2
        assert lib.mock_live_bytes() == live0
        # a stock allocation is one call, as before
        _malloc(lib, "pm_stock", 6 * MIB)
        _malloc(lib, "pm_stock", 4 * MIB)
        assert _pause(lib, "pm_stock") == 2
        assert _sub(lib)[2:] == (2, 2) and _maps(lib)[2:5] == (0, 0, 0)
        assert lib.mock_live_bytes() == live0
    finally:
        lib.tms_set_pause_coalesce(0)


def test_on_driver_refuses_the_range_falls_back_per_extent(lib):
    lib.tms_set_pause_coalesce(1)
    lib.mock_set_unmap_multi(0)
    try:
        live0 = lib.mock_live_bytes()
        _lattice(lib, "pm_refused")
        # the refused run call, then its three extents, then the lone one
        assert _pause(lib, "pm_refused") == 5
        assert _sub(lib)[1:] == ("pm_refused", 1, 5)
        assert _maps(lib)[1:] == ("pm_refused", 4, 0, 1, 1)
        assert lib.mock_live_bytes() == live0
        assert lib.tms_resume_rc(b"pm_refused") == 0
        assert lib.mock_live_bytes() - live0 == 12 * MIB
    finally:
        lib.mock_set_unmap_multi(1)
        lib.tms_set_pause_coalesce(0)
    lib.tms_pause(b"pm_refused")


def test_unsorted_extents_after_a_live_grow_are_one_run(lib):
    """A live grow appends the new cells BEHIND the kept ones in the saver's
    list; the run is cut on offsets, not on list order."""
    live0 = lib.mock_live_bytes()
    p = _malloc(lib, "pm_grow", 16 * MIB)
    lo = (ctypes.c_uint64 * 2)(4 * MIB, 6 * MIB)
    hi = (ctypes.c_uint64 * 2)(6 * MIB, 8 * MIB)
    assert lib.tms_set_spans(ctypes.c_void_p(p), 2, lo, hi, 1) == 0
    lo2 = (ctypes.c_uint64 * 3)(0, 4 * MIB, 6 * MIB)
    hi2 = (ctypes.c_uint64 * 3)(4 * MIB, 6 * MIB, 8 * MIB)
    assert lib.tms_set_spans(ctypes.c_void_p(p), 3, lo2, hi2, 1) == 0  # grow: [0,4) mapped last
    lib.tms_set_pause_coalesce(1)
    try:
        assert _pause(lib, "pm_grow") == 1
        assert _maps(lib)[1:] == ("pm_grow", 3, 1, 0, 1)
        assert lib.mock_live_bytes() == live0
    finally:
        lib.tms_set_pause_coalesce(0)


# ---- (2) the switch -------------------------------------------------------------------

def test_switch_default_on_after_metal():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_ENABLE_PAUSE_COALESCE_UNMAP.get() is True   # y4l metal


def test_arm_pushes_the_switch_and_names_a_missing_hook(monkeypatch, caplog):
    from sglang.srt.weg2 import pause_maps as pm

    seen = []
    ad = types.SimpleNamespace(set_pause_coalesce=lambda on: seen.append(on) or on)
    monkeypatch.setattr(pm, "_last", None)
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_PAUSE_COALESCE_UNMAP", "0")
    with caplog.at_level(logging.INFO):
        assert pm.arm(ad) is False
    assert seen == [False] and "WEG2-PAUSE-MAPS coalesce=0" in caplog.text
    caplog.clear()
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_PAUSE_COALESCE_UNMAP", "1")
    with caplog.at_level(logging.INFO):
        assert pm.arm(ad) is True
        assert pm.arm(ad) is True
    assert seen == [False, True, True]
    assert caplog.text.count("WEG2-PAUSE-MAPS coalesce=1") == 1   # logged on change only
    caplog.clear()
    monkeypatch.setattr(pm, "_last", None)
    with caplog.at_level(logging.INFO):
        assert pm.arm(types.SimpleNamespace()) is None
    assert "has no tms_set_pause_coalesce" in caplog.text


def test_arm_against_the_built_saver(lib, monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tma
    from sglang.srt.weg2 import pause_maps as pm

    monkeypatch.setattr(tma, "_weg2_ring_symbol", lambda name: getattr(lib, name, None))
    ad = object.__new__(tma._TorchMemorySaverAdapterReal)
    monkeypatch.setattr(pm, "_last", None)
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_PAUSE_COALESCE_UNMAP", "1")
    try:
        assert pm.arm(ad) is True
        _lattice(lib, "pm_armed")
        assert _pause(lib, "pm_armed") == 2
        st = ad.pause_maps_stats("pm_armed")
        assert st == {"extents": 4, "runs": 1, "fallbacks": 0, "coalesce": True}
        assert ad.pause_maps_stats("another") is None
        assert ad.pause_stats("pm_armed")["unmaps"] == 2
    finally:
        monkeypatch.setenv("SGLANG_WEG2_ENABLE_PAUSE_COALESCE_UNMAP", "0")
        assert pm.arm(ad) is False
    monkeypatch.setattr(tma, "_weg2_ring_symbol", lambda name: None)
    assert ad.set_pause_coalesce(True) is None and ad.pause_maps_stats("pm_armed") is None
    noop = tma._TorchMemorySaverAdapterNoop()
    assert noop.set_pause_coalesce(True) is None and noop.pause_maps_stats("x") is None


# ---- (3) the line and the wiring -----------------------------------------------------

def test_pause_sub_line_carries_the_extent_census(caplog):
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    stub = types.SimpleNamespace(memory_saver_adapter=types.SimpleNamespace(
        pause_stats=lambda t: {"allocations": 10, "unmaps": 10, "unmap_ms": 11.0,
                               "release_ms": 2.0, "total_ms": 13.1},
        pause_maps_stats=lambda t: {"extents": 54, "runs": 6, "fallbacks": 0, "coalesce": True}))
    with caplog.at_level(logging.INFO):
        wu.SchedulerWeightUpdaterManager._weg2_pause_sub_line(stub, "weights_3", 13.2)
    assert ("WEG2-PAUSE-SUB tag=weights_3 allocs=10 unmaps=10 unmap_ms=11.0 release_ms=2.0 "
            "native_ms=13.1 pause_ms=13.2 extents=54 runs=6 fallbacks=0 coalesce=1") in caplog.text
    caplog.clear()
    stub.memory_saver_adapter.pause_maps_stats = lambda t: None       # older hook: no suffix
    with caplog.at_level(logging.INFO):
        wu.SchedulerWeightUpdaterManager._weg2_pause_sub_line(stub, "weights_3", 13.2)
    assert "pause_ms=13.2 (unmaps" in caplog.text


def test_the_sleep_leg_arms_the_saver_before_its_loop():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = open(wu.__file__).read()
    i = src.index("_weg2_pause_maps.arm(self.memory_saver_adapter)")
    j = src.index("self._weg2_h111b_scope(recv_req, weights_tags) as _h111b, \\")
    k = src.index("def _weg2_pause_step(")
    assert i < j < k and j - i < 800
