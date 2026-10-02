# SPDX-License-Identifier: Apache-2.0
"""L15-EXTENTS (N3y 08:41:37Z "export of extent @0 refused rc=-2"): the hold
share and the keep arm trust the saver's span extents (tms_list_extents),
not l15_keep_split's memory of the split."""

from __future__ import annotations

import ctypes

import pytest

from sglang.srt.weg2 import l15_keep_split as ks
from sglang.srt.weg2.l15_hold_share import (
    L15ShareError,
    export_hold_extents,
    list_extents,
    native_cover,
)

M = 1 << 20


def _lister(extents):
    def fn(ptr, cap, offs, sizes):
        for i, (o, z) in enumerate(extents[: int(cap.value)]):
            offs[i] = o
            sizes[i] = z
        return len(extents)
    return fn


def _exporter(calls):
    def fn(ptr, off, fd, size):
        calls.append(int(off.value))
        fd._obj.value = 100 + len(calls)
        size._obj.value = 2 * M
        return 0
    return fn


def test_native_cover_exact_finer_and_missing():
    assert native_cover([(0, 4 * M), (4 * M, 16 * M)], [(0, 4 * M)]) == [(0, 4 * M)]
    assert native_cover([(0, 2 * M), (2 * M, 2 * M), (4 * M, 4 * M)],
                        [(0, 4 * M)]) == [(0, 2 * M), (2 * M, 2 * M)]
    assert native_cover([], [(0, 4 * M)]) is None             # stock
    assert native_cover([(0, 8 * M)], [(0, 4 * M)]) is None    # straddles
    assert native_cover([(2 * M, 2 * M)], [(0, 4 * M)]) is None  # hole at 0


def test_list_extents_reads_the_saver():
    assert list_extents(0x1000, _lister([(0, 2 * M), (2 * M, 6 * M)])) == [
        (0, 2 * M), (2 * M, 6 * M)]
    assert list_extents(0x1000, lambda *a: -1) is None


def test_export_takes_every_native_extent_of_the_region():
    calls = []
    got = export_hold_extents(
        0x1000, [(0, 4 * M)], export=_exporter(calls),
        lister=_lister([(0, 2 * M), (2 * M, 2 * M), (4 * M, 12 * M)]))
    assert calls == [0, 2 * M] and [g[0] for g in got] == [0, 2 * M]


def test_export_refuses_a_region_the_saver_does_not_cover_by_name():
    with pytest.raises(L15ShareError) as ei:
        export_hold_extents(0x1000, [(0, 4 * M)], export=_exporter([]),
                            lister=_lister([]))
    assert "do not cover the hold regions" in str(ei.value)


def test_keep_arm_refuses_when_the_saver_lost_the_split():
    ks.forget_all()
    ks._HOLD[0x2000] = ((0, 4 * M),)
    assert ks.keep_extents(0x2000, [(0, M)], native=[(0, 4 * M), (4 * M, 4 * M)]) == [(0, 4 * M)]
    assert ks.keep_extents(0x2000, [(0, M)], native=[]) is None
    assert ks.keep_extents(0x2000, [], native=[]) == []
    ks.forget_all()


def test_native_list_extents_against_the_mock_driver(tmp_path):
    """The real core.cpp built against the h95c mock driver: a stock base
    lists [], a split base its plan, a pause keeps only the kept extents."""
    import importlib.util
    import os
    import shutil
    import subprocess
    import textwrap

    gxx = shutil.which("g++")
    inc = "/usr/local/cuda/include"
    if gxx is None or not os.path.isfile(os.path.join(inc, "cuda.h")):
        pytest.skip("no g++ / CUDA headers")
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        "h95c", os.path.join(here, "test_weg2_d_seat_vram_h95c.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    import sglang.srt.weg2 as w

    src = os.path.join(os.path.dirname(w.__file__), "tms_csrc")
    (tmp_path / "mock.cpp").write_text(textwrap.dedent(h._MOCK))
    so = tmp_path / "libtms.so"
    r = subprocess.run([gxx, "-std=c++17", "-shared", "-fPIC", "-DUSE_CUDA",
                        "-DTMS_HOOK_MODE_PRELOAD", "-I" + inc, "-I" + src]
                       + [os.path.join(src, f) for f in ("core.cpp", "entrypoint.cpp",
                                                         "host_ring.cpp", "api_forwarder.cpp")]
                       + [str(tmp_path / "mock.cpp"), "-o", str(so), "-Wl,-Bsymbolic",
                          "-ldl", "-lpthread"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    lib = ctypes.CDLL(str(so))
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    lib.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    lib.tms_pause.argtypes = [ctypes.c_char_p]
    lib.tms_list_extents.restype = ctypes.c_int
    lib.tms_list_extents.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_uint64),
                                     ctypes.POINTER(ctypes.c_uint64)]
    lib.tms_set_keep_spans.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.POINTER(ctypes.c_uint64),
                                       ctypes.POINTER(ctypes.c_uint64)]
    from sglang.srt.weg2 import d_seat_vram as dsv

    spans = dsv.TmsSpans(symbol=lambda n: getattr(lib, n))
    lib.tms_set_current_tag(b"kvx")
    lib.tms_set_interesting_region(True)
    p = ctypes.c_void_p()
    assert lib.cudaMalloc(ctypes.byref(p), 20 * M) == 0
    lib.tms_set_interesting_region(False)
    p = int(p.value)
    lister = lib.tms_list_extents
    assert list_extents(p, lister) == []                       # stock
    assert list_extents(p + 2 * M, lister) is None             # not a base
    assert spans.set_spans(p, [(0, 4 * M), (4 * M, 20 * M)], now=True) == 0
    assert list_extents(p, lister) == [(0, 4 * M), (4 * M, 16 * M)]
    lo = (ctypes.c_uint64 * 1)(0)
    hi = (ctypes.c_uint64 * 1)(4 * M)
    assert lib.tms_set_keep_spans(ctypes.c_void_p(p), 1, lo, hi) == 0
    lib.tms_pause(b"kvx")
    assert list_extents(p, lister) == [(0, 4 * M)]
