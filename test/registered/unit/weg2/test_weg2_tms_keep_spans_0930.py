# SPDX-License-Identifier: Apache-2.0
"""TMS patch 6 (01.10., AP L15-03): the KEEP SPANS -- ranges that survive the pause.

L1.5 holds D's KV prefix through the P phase (plan 2.2/5): a paused D rank
keeps the kept byte ranges of a span-mapped allocation mapped -- their pages
are neither unmapped nor released -- while every other extent goes back the
patch-5 way.  The resume maps only the gaps of the plan; kept extents keep
their handle, so their content (and every captured graph's data) survives.

The mock driver cannot hold content, so the mock's handle counter stands in
for content: cuMemCreate hands out a FRESH handle per call, so a kept extent
that survived the cycle still carries its old handle, while a re-mapped gap
comes back with a different one.

Fixture pattern (mock build, lattice, counters) copied from
test_weg2_pause_maps_0930.py; that file and test_weg2_d_seat_vram_h95c.py
are NOT modified -- the mock is extended through the same source-rewrite
trick (anchor-checked).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import subprocess
import textwrap

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

_ERR_ANCHOR = 'const char* cudaGetErrorString(cudaError_t) { return "mock"; }'
_HANDLE_AT = (
    _ERR_ANCHOR + "\n"
    "long long mock_handle_at(CUdeviceptr a) {"
    "  auto it = g_map.find(a);"
    "  return it == g_map.end() ? -1 : (long long) it->second.second; }"
)


def _mock_source() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_h95c_for_keep_spans", os.path.join(here, "test_weg2_d_seat_vram_h95c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = textwrap.dedent(mod._MOCK)
    assert _UNMAP_STOCK in src, "the H95c mock's cuMemUnmap changed -- update this test"
    assert _ERR_ANCHOR in src, "the H95c mock's error-string hook changed -- update this test"
    src = src.replace("#include <map>", "#include <map>\n#include <vector>")
    src = src.replace(_UNMAP_STOCK, _UNMAP_MULTI)
    return src.replace(_ERR_ANCHOR, _HANDLE_AT)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    gxx = shutil.which("g++")
    inc = "/usr/local/cuda/include"
    if gxx is None or not os.path.isfile(os.path.join(inc, "cuda.h")):
        pytest.skip("no g++ / CUDA headers to build the saver against a mock driver")
    import sglang.srt.weg2 as w

    src = os.path.join(os.path.dirname(w.__file__), "tms_csrc")
    out = tmp_path_factory.mktemp("tms_keep_spans")
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
    lib.tms_set_keep_spans.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
                                       ctypes.POINTER(ctypes.c_uint64)]
    lib.tms_set_pause_coalesce.argtypes = [ctypes.c_int]
    lib.tms_tag_bytes.argtypes = [ctypes.c_char_p]
    lib.tms_tag_bytes.restype = ctypes.c_uint64
    lib.tms_tag_mapped_bytes.argtypes = [ctypes.c_char_p]
    lib.tms_tag_mapped_bytes.restype = ctypes.c_uint64
    lib.tms_alloc_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                                   ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                                   ctypes.POINTER(ctypes.c_int)]
    lib.mock_live_bytes.restype = ctypes.c_size_t
    lib.mock_mapped_bytes.restype = ctypes.c_size_t
    lib.mock_unmap_calls.restype = ctypes.c_long
    lib.mock_extents.restype = ctypes.c_int
    lib.mock_is_mapped.argtypes = [ctypes.c_void_p]
    lib.mock_handle_at.argtypes = [ctypes.c_void_p]
    lib.mock_handle_at.restype = ctypes.c_longlong
    return lib


def _malloc(lib, tag, size):
    lib.tms_set_current_tag(tag.encode())
    lib.tms_set_interesting_region(True)
    p = ctypes.c_void_p()
    assert lib.cudaMalloc(ctypes.byref(p), size) == 0
    lib.tms_set_interesting_region(False)
    return int(p.value)


def _ranges(vals):
    lo = (ctypes.c_uint64 * len(vals))(*[v for v, _ in vals])
    hi = (ctypes.c_uint64 * len(vals))(*[v for _, v in vals])
    return lo, hi


def _lattice(lib, tag):
    """One 16 MiB allocation, span-mapped NOW into four extents:
    [0,4) [4,6) [6,8) [10,14) MiB -- the H95c lattice shape."""
    p = _malloc(lib, tag, 16 * MIB)
    lo, hi = _ranges([(0, 4 * MIB), (4 * MIB, 6 * MIB), (6 * MIB, 8 * MIB), (10 * MIB, 14 * MIB)])
    assert lib.tms_set_spans(ctypes.c_void_p(p), 4, lo, hi, 1) == 0
    return p


def _set_keep(lib, p, vals):
    lo, hi = _ranges(vals)
    return lib.tms_set_keep_spans(ctypes.c_void_p(p), len(vals), lo, hi)


def _pause_unmaps(lib, tag):
    c0 = lib.mock_unmap_calls()
    lib.tms_pause(tag.encode())
    return lib.mock_unmap_calls() - c0


def _resume(lib, tag):
    return lib.tms_resume_rc(tag.encode())


def _handles(lib, p):
    return [lib.mock_handle_at(ctypes.c_void_p(p + off))
            for off in (0, 4 * MIB, 6 * MIB, 10 * MIB)]


def test_keep_spans_survive_pause_resume_maps_only_gaps(lib):
    """(a)+(b): keep covers extents 0-1.  The pause unmaps only extents 2-3
    and leaves 6 MiB mapped; the resume maps the two gaps with FRESH handles
    while the kept handles stay the originals."""
    p = _lattice(lib, "l15ka")
    h = _handles(lib, p)
    assert all(v > 0 for v in h)
    mapped0, live0 = lib.mock_mapped_bytes(), lib.mock_live_bytes()

    assert _set_keep(lib, p, [(0, 6 * MIB)]) == 0
    assert _pause_unmaps(lib, "l15ka") == 2                    # extents 2-3 only
    assert lib.mock_mapped_bytes() == mapped0 - 6 * MIB        # kept: extents 0-1's bytes
    assert lib.mock_live_bytes() == live0 - 6 * MIB            # the gone handles released
    assert lib.mock_handle_at(ctypes.c_void_p(p)) == h[0]      # kept handle unchanged
    assert lib.mock_handle_at(ctypes.c_void_p(p + 4 * MIB)) == h[1]
    assert not lib.mock_is_mapped(ctypes.c_void_p(p + 7 * MIB))
    assert not lib.mock_is_mapped(ctypes.c_void_p(p + 12 * MIB))

    assert _resume(lib, "l15ka") == 0
    assert lib.mock_handle_at(ctypes.c_void_p(p)) == h[0]      # not re-mapped
    assert lib.mock_handle_at(ctypes.c_void_p(p + 4 * MIB)) == h[1]
    n2 = lib.mock_handle_at(ctypes.c_void_p(p + 6 * MIB))      # the gaps came back fresh
    n3 = lib.mock_handle_at(ctypes.c_void_p(p + 10 * MIB))
    assert n2 not in (-1, h[2]) and n3 not in (-1, h[3])
    assert lib.mock_mapped_bytes() == mapped0
    assert lib.mock_live_bytes() == live0


def test_empty_keep_set_is_the_patch5_walk(lib):
    """(c) golden: an empty keep set -- never set, or cleared with n == 0 --
    walks EXACTLY the patch-5 pause/resume: 4 one-by-one unmaps, everything
    back on resume, same counters in both cycles."""
    p = _lattice(lib, "l15kb")
    mapped0 = lib.mock_mapped_bytes()

    base_pause = _pause_unmaps(lib, "l15kb")
    assert base_pause == 4                                     # one unmap per extent
    assert lib.mock_mapped_bytes() == mapped0 - 12 * MIB       # all four extents gone
    assert _resume(lib, "l15kb") == 0

    assert _set_keep(lib, p, []) == 0                          # n == 0 clears the keep set
    assert _pause_unmaps(lib, "l15kb") == base_pause
    assert lib.mock_mapped_bytes() == mapped0 - 12 * MIB
    assert _resume(lib, "l15kb") == 0


def test_keep_applies_inside_the_coalesced_branch(lib):
    """Patch 5's coalescing sees ONLY the non-kept extents: keeping extent 0
    leaves the run [4,8) (one cuMemUnmap) plus the lone [10,14) (fallback) --
    two unmap calls, not the full lattice's."""
    p = _lattice(lib, "l15kc")
    h = _handles(lib, p)
    assert _set_keep(lib, p, [(0, 4 * MIB)]) == 0
    assert lib.tms_set_pause_coalesce(1) == 1
    try:
        assert _pause_unmaps(lib, "l15kc") == 2                # run [4,8) + lone [10,14)
    finally:
        assert lib.tms_set_pause_coalesce(0) == 0
    assert lib.mock_handle_at(ctypes.c_void_p(p)) == h[0]      # kept extent untouched
    assert _resume(lib, "l15kc") == 0
    assert lib.mock_handle_at(ctypes.c_void_p(p)) == h[0]


def test_keep_span_refusals(lib):
    """(d) the same refusals as set_spans: -1 not an allocation base, -3 a
    malformed plan (unsorted / overlapping / outside / unaligned).  A refused
    call leaves the keep set untouched."""
    p = _lattice(lib, "l15kd")
    lo, hi = _ranges([(0, 2 * MIB)])
    assert lib.tms_set_keep_spans(ctypes.c_void_p(p + MIB), 1, lo, hi) == -1
    assert _set_keep(lib, p, [(4 * MIB, 6 * MIB), (0, 4 * MIB)]) == -3    # unsorted
    assert _set_keep(lib, p, [(0, 6 * MIB), (4 * MIB, 8 * MIB)]) == -3    # overlapping
    assert _set_keep(lib, p, [(14 * MIB, 18 * MIB)]) == -3                # outside the 16 MiB
    assert _set_keep(lib, p, [(1 * MIB, 3 * MIB)]) == -3                  # 2 MiB granularity

    assert _set_keep(lib, p, [(0, 6 * MIB)]) == 0              # valid keep
    assert _set_keep(lib, p, [(8 * MIB, 9 * MIB), (0, 4 * MIB)]) == -3    # refused mid-call
    assert _pause_unmaps(lib, "l15kd") == 2                    # the valid keep still holds
    assert _resume(lib, "l15kd") == 0


def _mapped(lib, p):
    size = ctypes.c_uint64()
    mapped = ctypes.c_uint64()
    planned = ctypes.c_uint64()
    active = ctypes.c_int()
    rc = lib.tms_alloc_info(ctypes.c_void_p(p), ctypes.byref(size), ctypes.byref(mapped),
                            ctypes.byref(planned), ctypes.byref(active))
    assert rc == 0
    return int(mapped.value)


def test_paused_keep_spans_count_as_mapped(lib):
    """L15-13a: a paused allocation that holds KEEP SPANS reports the kept
    bytes as mapped -- alloc_info's mapped and tms_tag_mapped_bytes answer
    "physical bytes mapped NOW" -- while tms_tag_bytes keeps its H95c seat
    semantics (mapped now or planned for the next resume)."""
    p = _lattice(lib, "l15ke")
    mapped_active = _mapped(lib, p)
    plan = lib.tms_tag_bytes(b"l15ke")
    print(f"[l15-13a] ACTIVE: mapped={mapped_active} tag_bytes={plan} "
          f"tag_mapped={lib.tms_tag_mapped_bytes(b'l15ke')}")
    assert mapped_active == 12 * MIB and plan == 12 * MIB
    assert _set_keep(lib, p, [(0, 6 * MIB)]) == 0
    _pause_unmaps(lib, "l15ke")
    mapped_paused = _mapped(lib, p)
    print(f"[l15-13a] PAUSED: mapped={mapped_paused} tag_bytes={lib.tms_tag_bytes(b'l15ke')} "
          f"tag_mapped={lib.tms_tag_mapped_bytes(b'l15ke')}")
    assert mapped_paused == 6 * MIB
    assert lib.tms_tag_mapped_bytes(b"l15ke") == 6 * MIB
    assert lib.tms_tag_bytes(b"l15ke") == plan        # unchanged: the resume plan
    assert _resume(lib, "l15ke") == 0
    assert _mapped(lib, p) == mapped_active


def test_paused_without_keep_reads_zero(lib):
    """L15-13a: a paused allocation with an EMPTY keep set keeps reading zero
    mapped -- byte-identical to the pre-fix accounting for keep-less paths."""
    p = _lattice(lib, "l15kf")
    assert _set_keep(lib, p, []) == 0
    _pause_unmaps(lib, "l15kf")
    assert _mapped(lib, p) == 0
    assert lib.tms_tag_mapped_bytes(b"l15kf") == 0
    assert _resume(lib, "l15kf") == 0


def _info(lib, p):
    size = ctypes.c_uint64()
    mapped = ctypes.c_uint64()
    planned = ctypes.c_uint64()
    active = ctypes.c_int()
    rc = lib.tms_alloc_info(ctypes.c_void_p(p), ctypes.byref(size), ctypes.byref(mapped),
                            ctypes.byref(planned), ctypes.byref(active))
    assert rc == 0
    return (int(size.value), int(mapped.value), int(planned.value), int(active.value))


def test_keep_span_accounting_consistent_over_a_cycle(lib):
    """L15-D4: end-to-end accounting over ONE full cycle -- active, paused
    with a keep span, resumed, paused again with the keep cleared.  At every
    point the three readers of "physical bytes mapped NOW" agree:
    tms_tag_mapped_bytes == alloc_info's mapped == the mock driver's mapped
    bytes of THIS allocation (a difference against the global mock counter,
    the mock is module-wide).  While paused, tms_tag_bytes keeps the H95c
    seat semantics and equals alloc_info's planned."""
    g0 = lib.mock_mapped_bytes()                    # mapped bytes NOT from my allocation
    p = _lattice(lib, "l15kg")
    tag = b"l15kg"
    plan = 12 * MIB                                 # spans [0,4) [4,6) [6,8) [10,14) MiB
    keep = 6 * MIB                                  # extents 0-1 survive the pause

    def snap():
        size, mapped, planned, active = _info(lib, p)
        return (size, mapped, planned, active,
                lib.tms_tag_bytes(tag), lib.tms_tag_mapped_bytes(tag),
                lib.mock_mapped_bytes() - g0)

    # (1) active before the pause: the whole plan mapped, all readers agree.
    size, mapped, planned, active, tag_bytes, tag_mapped, mock_mapped = snap()
    assert size == 16 * MIB and mapped == plan and planned == plan and active == 1
    assert tag_bytes == tag_mapped == mock_mapped == plan

    # (2) paused with the keep span: the kept 6 MiB stay mapped and are
    # counted by every reader; the seat budget still answers the full plan.
    assert _set_keep(lib, p, [(0, keep)]) == 0
    assert _pause_unmaps(lib, "l15kg") == 2         # extents 2-3 only
    size, mapped, planned, active, tag_bytes, tag_mapped, mock_mapped = snap()
    assert size == 16 * MIB and active == 0
    assert mapped == tag_mapped == mock_mapped == keep
    assert planned == plan and tag_bytes == planned  # H95c semantics while paused
    kept_handles = _handles(lib, p)[:2]              # extents 0-1 are the kept ones
    assert all(v > 0 for v in kept_handles)

    # (3) resume: the plan is mapped again in full (16 MiB is the allocation
    # size; the span plan maps 12 MiB of it) and the kept extents carry
    # their ORIGINAL handles -- the gaps came back with fresh ones.
    assert _resume(lib, "l15kg") == 0
    size, mapped, planned, active, tag_bytes, tag_mapped, mock_mapped = snap()
    assert size == 16 * MIB and active == 1
    assert mapped == tag_mapped == mock_mapped == plan
    assert tag_bytes == plan
    assert _handles(lib, p)[:2] == kept_handles     # kept handles survived

    # (4) keep cleared, paused again: the patch-5 walk -- nothing stays
    # mapped, yet the seat budget still names the resume plan.
    assert _set_keep(lib, p, []) == 0
    assert _pause_unmaps(lib, "l15kg") == 4         # all four extents go
    size, mapped, planned, active, tag_bytes, tag_mapped, mock_mapped = snap()
    assert size == 16 * MIB and active == 0
    assert mapped == tag_mapped == mock_mapped == 0
    assert planned == plan and tag_bytes == planned
