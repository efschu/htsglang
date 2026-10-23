"""BAR1 collector: ONE runtime call for N device-to-device copies (1023).

``CudartBatchCopier`` issues a batch of flat copies as one
``cudaMemcpyBatchAsync`` and a batch of strided-2D copies as one
``cudaMemcpy3DBatchAsync`` (ctypes.PyDLL, GIL held). A ctypes layout mistake
would not fail loudly: the runtime would read wrong pointers/pitches and copy
weights to the wrong place. Pinned here, no GPU anywhere:

 1. LAYOUT against the REAL header: gcc compiles a program over
    ``cuda_runtime_api.h`` that prints sizeof/offsetof of every mirrored
    struct; ctypes must agree (the load-bearing test).
 2. MARSHALING with a fake lib that captures the arguments (13 form, 12.x form
    with ``failIdx``).
 3. CANARY logic with a fake lib whose "device memory" is host memory: right
    emulation -> handed out; one copy misplaced by one byte / rc != 0 /
    runtime 12070 / missing symbol -> refused with a named reason.
 4. DeviceOps base and desk fakes without the method -> (None, named why).
 5. END TO END through the real ctypes marshaling: a gcc-built stub libcudart
    (header structs, API-contract checks on every argument) loaded by the real
    ``CudartDeviceOps`` -- the C ABI reads what ``prepare`` wrote.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402

@pytest.fixture(autouse=True)
def _fresh_copier_cache():
    """The verdict is cached per process (library path, device); every test
    starts without one."""
    tp.reset_batch_copier_cache()
    yield
    tp.reset_batch_copier_cache()


CU13_INCLUDE = ("/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/"
                "nvidia/cu13/include")
GCC = "/bin/gcc" if os.path.exists("/bin/gcc") else (shutil.which("gcc") or "")


def _gcc_skip_reason() -> str:
    if not GCC:
        return "no gcc on this desk"
    if not os.path.exists(os.path.join(CU13_INCLUDE, "cuda_runtime_api.h")):
        return f"cu13 headers not found under {CU13_INCLUDE}"
    return ""


# ---------------------------------------------------------------------------
# 1. layout against the real header
# ---------------------------------------------------------------------------

# (C type, ctypes class, [field paths]); a path walks ctypes fields exactly as
# C walks members (unions are offset 0 per member).
LAYOUT = [
    ("struct cudaMemLocation", tp._CuMemLocation, ["type", "id"]),
    ("struct cudaMemcpyAttributes", tp._CuMemcpyAttributes,
     ["srcAccessOrder", "srcLocHint", "dstLocHint", "flags"]),
    ("struct cudaExtent", tp._CuExtent, ["width", "height", "depth"]),
    ("struct cudaOffset3D", tp._CuOffset3D, ["x", "y", "z"]),
    ("struct cudaMemcpy3DOperand", tp._CuMemcpy3DOperand,
     ["type", "op", "op.ptr.ptr", "op.ptr.rowLength", "op.ptr.layerHeight",
      "op.ptr.locHint", "op.array.array", "op.array.offset"]),
    ("struct cudaMemcpy3DBatchOp", tp._CuMemcpy3DBatchOp,
     ["src", "dst", "extent", "srcAccessOrder", "flags"]),
]


def _ctypes_offset(cls, path: str) -> int:
    off = 0
    for part in path.split("."):
        fld = getattr(cls, part)          # CField descriptor
        off += fld.offset
        cls = dict((n, t) for n, t, *_ in cls._fields_)[part]
    return off


def _compile_layout_probe(tmp_path):
    lines = ['#include <stdio.h>', '#include <stddef.h>',
             '#include <cuda_runtime_api.h>', 'int main(void) {']
    for ctype, _cls, paths in LAYOUT:
        lines.append(f'printf("sizeof|{ctype}|%zu\\n", sizeof({ctype}));')
        for p in paths:
            lines.append(
                f'printf("offsetof|{ctype}|{p}|%zu\\n", offsetof({ctype}, {p}));')
    lines.append('printf("const|SRC_ACCESS_ORDER_STREAM|%d\\n", '
                 '(int)cudaMemcpySrcAccessOrderStream);')
    lines.append('printf("const|OPERAND_TYPE_POINTER|%d\\n", '
                 '(int)cudaMemcpyOperandTypePointer);')
    lines.append('return 0; }')
    src = tmp_path / "layout.c"
    src.write_text("\n".join(lines))
    exe = tmp_path / "layout"
    proc = subprocess.run([GCC, f"-I{CU13_INCLUDE}", str(src), "-o", str(exe)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("layout probe does not compile against the header "
                    f"(named reason): {proc.stderr[-600:]}")
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    return [ln.split("|") for ln in out.stdout.splitlines()]


def test_layout_matches_real_header(tmp_path):
    reason = _gcc_skip_reason()
    if reason:
        pytest.skip(reason)
    rows = _compile_layout_probe(tmp_path)
    classes = {ct: cls for ct, cls, _ in LAYOUT}
    seen_sizes = seen_offsets = 0
    for row in rows:
        if row[0] == "sizeof":
            _, ctype, n = row
            assert ctypes.sizeof(classes[ctype]) == int(n), (ctype, n)
            seen_sizes += 1
        elif row[0] == "offsetof":
            _, ctype, path, n = row
            assert _ctypes_offset(classes[ctype], path) == int(n), (ctype, path, n)
            seen_offsets += 1
        else:
            _, name, val = row
            const = {"SRC_ACCESS_ORDER_STREAM": tp.CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM,
                     "OPERAND_TYPE_POINTER": tp.CUDA_MEMCPY_OPERAND_TYPE_POINTER}[name]
            assert const == int(val), name
    assert seen_sizes == len(LAYOUT)
    assert seen_offsets == sum(len(p) for _, _, p in LAYOUT)


def test_layout_numbers_pinned_without_gcc():
    # the cu13 numbers the module comment quotes; independent of gcc so a
    # skipped compile cannot hide a drifted mirror
    assert ctypes.sizeof(tp._CuMemLocation) == 8
    assert ctypes.sizeof(tp._CuMemcpyAttributes) == 24
    assert ctypes.sizeof(tp._CuMemcpy3DOperand) == 40
    assert ctypes.sizeof(tp._CuExtent) == 24
    assert ctypes.sizeof(tp._CuMemcpy3DBatchOp) == 112
    assert _ctypes_offset(tp._CuMemcpy3DBatchOp, "dst") == 40
    assert _ctypes_offset(tp._CuMemcpy3DBatchOp, "extent") == 80
    assert _ctypes_offset(tp._CuMemcpy3DBatchOp, "srcAccessOrder") == 104
    assert _ctypes_offset(tp._CuMemcpy3DBatchOp, "flags") == 108
    assert _ctypes_offset(tp._CuMemcpy3DOperand, "op.ptr.rowLength") == 16


# ---------------------------------------------------------------------------
# a fake runtime whose "device memory" is host memory
# ---------------------------------------------------------------------------


class FakeRuntime:
    """``lib`` (the CDLL half) and ``held`` (the PyDLL half) of a fake libcudart.

    Device allocations are slices of one host arena, addressed by their REAL
    host address, so every copy is a memmove. ``captured`` records the raw
    arguments of every batch call."""

    def __init__(self, version=13000, rc=0, misplace=False, missing=()):
        self.version = version
        self.rc = rc
        self.misplace = misplace
        self.arena = ctypes.create_string_buffer(1 << 20)
        self.top = ctypes.addressof(self.arena)
        self.end = self.top + (1 << 20)
        self.captured = []
        self.freed = []
        self.live = set()
        self.streams_made = 0
        self.streams_destroyed = 0
        self.last_error_cleared = 0
        self.synced = []
        outer = self

        def _obj(ref):
            return ref._obj

        def malloc(ref, n):
            addr = (outer.top + 255) & ~255
            if addr + n > outer.end:
                return 2
            outer.top = addr + n
            _obj(ref).value = addr
            outer.live.add(addr)
            return 0

        def free(ptr):
            outer.freed.append(ptr.value)
            outer.live.discard(ptr.value)
            return 0

        def memcpy(dst, src, n, kind):
            ctypes.memmove(dst.value, src.value, n)
            return 0

        def stream_create(ref):
            outer.streams_made += 1
            _obj(ref).value = 0x1000 + outer.streams_made
            return 0

        def stream_flags(stream, ref):
            _obj(ref).value = 0
            return 0

        def stream_destroy(stream):
            outer.streams_destroyed += 1
            return 0

        def stream_sync(stream):
            outer.synced.append(stream.value)
            return 0

        def get_version(ref):
            _obj(ref).value = outer.version
            return 0

        def get_last_error():
            outer.last_error_cleared += 1
            return 0

        self.lib = SimpleNamespace(
            cudaGetErrorString=lambda rc: b"fake cuda error %d" % rc,
            cudaGetLastError=get_last_error,
            cudaSetDevice=lambda dev: 0,
            cudaStreamCreate=stream_create,
            cudaStreamGetFlags=stream_flags,
            cudaStreamDestroy=stream_destroy,
            cudaStreamSynchronize=stream_sync,
            cudaMalloc=malloc,
            cudaFree=free,
            cudaMemcpy=memcpy,
            cudaRuntimeGetVersion=get_version,
        )
        held = SimpleNamespace(
            cudaMemcpyBatchAsync=self._flat,
            cudaMemcpy3DBatchAsync=self._3d,
            cudaGetLastError=get_last_error,
        )
        # plain functions accept attribute assignment (argtypes), bound methods
        # do not: wrap
        held.cudaMemcpyBatchAsync = lambda *a: self._flat(*a)
        held.cudaMemcpy3DBatchAsync = lambda *a: self._3d(*a)
        for name in missing:
            delattr(held, name)
        self.held = held

    # -- batch emulation ----------------------------------------------------
    def _flat(self, *args):
        self.captured.append(("flat", args))
        if self.rc:
            return self.rc
        if len(args) == 9:
            dsts, srcs, sizes, count, attrs, idxs, nattrs, _fail, stream = args
        else:
            dsts, srcs, sizes, count, attrs, idxs, nattrs, stream = args
        for i in range(count):
            dst = dsts[i] + (1 if (self.misplace and i == 1) else 0)
            ctypes.memmove(dst, srcs[i], sizes[i])
        return 0

    def _3d(self, *args):
        self.captured.append(("3d", args))
        if self.rc:
            return self.rc
        if len(args) == 5:
            n, ops3d, _fail, flags, stream = args
        else:
            n, ops3d, flags, stream = args
        for k in range(n):
            op = ops3d[k]
            w, h = op.extent.width, op.extent.height
            dp = op.dst.op.ptr.rowLength or w
            sp = op.src.op.ptr.rowLength or w
            for r in range(h):
                ctypes.memmove(op.dst.op.ptr.ptr + r * dp,
                               op.src.op.ptr.ptr + r * sp, w)
        return 0

    def ops(self):
        o = tp.CudartDeviceOps.__new__(tp.CudartDeviceOps)
        o.path = "/fake/libcudart.so.13"
        o.lib = self.lib
        o.lib_held = self.held
        return o


def _host(n, seed=7):
    return ctypes.create_string_buffer(
        bytes(((i * 31 + seed) ^ (i >> 5)) & 0xFF for i in range(n)), n)


def _rows_for(src, dst):
    s, d = ctypes.addressof(src), ctypes.addressof(dst)
    return [
        (tp.FLAT, d + 3, s + 5, 100, 0, 0, 0),
        (tp.STRIDED2D, d + 2000, s + 1000, 40, 64, 90, 5),
        (tp.FLAT, d + 500, s + 700, 33, 0, 0, 0),
    ]


# ---------------------------------------------------------------------------
# 2. marshaling
# ---------------------------------------------------------------------------


def test_marshaling_13_form_one_call_per_part():
    rt = FakeRuntime(version=13000)
    ops = rt.ops()
    copier = tp.CudartBatchCopier(ops, 13000)
    src, dst = _host(4096), ctypes.create_string_buffer(4096)
    rows = _rows_for(src, dst)
    prep = copier.prepare(rows)
    assert (prep.n_flat, prep.n_2d, prep.calls, prep.copies) == (2, 1, 2, 3)
    copier.issue(prep, 0x77)
    kinds = [k for k, _ in rt.captured]
    assert kinds == ["flat", "3d"]          # exactly one call per part
    (_, fa), (_, ta) = rt.captured
    dsts, srcs, sizes, count, attrs, idxs, nattrs, stream = fa   # no failIdx
    s, d = ctypes.addressof(src), ctypes.addressof(dst)
    assert count == 2
    assert [dsts[i] for i in range(2)] == [d + 3, d + 500]
    assert [srcs[i] for i in range(2)] == [s + 5, s + 700]
    assert [sizes[i] for i in range(2)] == [100, 33]
    assert attrs.srcAccessOrder == 1 and attrs.flags == 0
    assert attrs.srcLocHint.type == 0 and attrs.dstLocHint.type == 0
    assert list(idxs) == [0] and nattrs == 1
    assert stream.value == 0x77
    n, ops3d, flags, stream3 = ta                                # no failIdx
    assert (n, flags, stream3.value) == (1, 0, 0x77)
    op = ops3d[0]
    assert op.src.type == 1 and op.dst.type == 1
    assert op.src.op.ptr.ptr == s + 1000 and op.src.op.ptr.rowLength == 90
    assert op.dst.op.ptr.ptr == d + 2000 and op.dst.op.ptr.rowLength == 64
    assert op.src.op.ptr.layerHeight == 0 and op.dst.op.ptr.layerHeight == 0
    assert (op.extent.width, op.extent.height, op.extent.depth) == (40, 5, 1)
    assert op.srcAccessOrder == 1 and op.flags == 0
    assert (copier.calls, copier.copies) == (2, 3)
    # and the bytes land where Python says
    for i in range(100):
        assert dst.raw[3 + i] == src.raw[5 + i]
    for r in range(5):
        assert dst.raw[2000 + r * 64:2000 + r * 64 + 40] == \
            src.raw[1000 + r * 90:1000 + r * 90 + 40]


def test_marshaling_12x_form_passes_fail_idx():
    rt = FakeRuntime(version=12080)
    copier = tp.CudartBatchCopier(rt.ops(), 12080)
    assert copier.fail_idx_form is True
    src, dst = _host(4096), ctypes.create_string_buffer(4096)
    copier.issue(copier.prepare(_rows_for(src, dst)), 5)
    (_, fa), (_, ta) = rt.captured
    assert len(fa) == 9 and len(ta) == 5
    assert isinstance(fa[7], ctypes.c_size_t) and fa[7].value == 0
    assert fa[8].value == 5
    assert isinstance(ta[2], ctypes.c_size_t) and ta[3] == 0 and ta[4].value == 5
    assert fa[6] == 1


def test_only_flat_or_only_2d_is_one_call_empty_is_none():
    rt = FakeRuntime()
    copier = tp.CudartBatchCopier(rt.ops(), 13000)
    src, dst = _host(4096), ctypes.create_string_buffer(4096)
    rows = _rows_for(src, dst)
    copier.issue(copier.prepare([rows[0], rows[2]]), 1)
    copier.issue(copier.prepare([rows[1]]), 1)
    copier.issue(copier.prepare([]), 1)
    assert [k for k, _ in rt.captured] == ["flat", "3d"]
    assert (copier.calls, copier.copies) == (2, 3)


def test_stream_zero_is_a_value_error_and_issues_nothing():
    rt = FakeRuntime()
    copier = tp.CudartBatchCopier(rt.ops(), 13000)
    src, dst = _host(4096), ctypes.create_string_buffer(4096)
    prep = copier.prepare(_rows_for(src, dst))
    with pytest.raises(ValueError, match="stream is 0"):
        copier.issue(prep, 0)
    assert rt.captured == [] and copier.calls == 0


def test_rc_nonzero_clears_the_latched_error_and_raises_with_text():
    rt = FakeRuntime(rc=1)
    copier = tp.CudartBatchCopier(rt.ops(), 13000)
    src, dst = _host(4096), ctypes.create_string_buffer(4096)
    with pytest.raises(RuntimeError, match=r"cudaMemcpyBatchAsync rc=1 fake cuda error 1"):
        copier.issue(copier.prepare(_rows_for(src, dst)), 3)
    assert rt.last_error_cleared == 1       # #1378 xsn71 house rule
    assert copier.calls == 0 and copier.copies == 0


@pytest.mark.parametrize("row, why", [
    ((tp.FLAT, 1, 2, 0, 0, 0, 0), "nbytes=0"),
    ((tp.STRIDED2D, 1, 2, 0, 8, 8, 2), "width=0"),
    ((tp.STRIDED2D, 1, 2, 8, 8, 8, 0), "rows=0"),
    ((tp.STRIDED2D, 1, 2, 16, 8, 16, 2), "dpitch=8"),
    ((tp.STRIDED2D, 1, 2, 16, 16, 8, 2), "spitch=8"),
    ((tp.ZEROFILL, 1, 2, 16, 0, 0, 0), "ZEROFILL"),
    ((tp.FLAT, 1, 2, 3), "4 fields"),
])
def test_prepare_refuses_bad_rows_by_name(row, why):
    copier = tp.CudartBatchCopier(FakeRuntime().ops(), 13000)
    with pytest.raises(ValueError, match=why):
        copier.prepare([row])


# ---------------------------------------------------------------------------
# 3. canary
# ---------------------------------------------------------------------------


def test_canary_good_emulation_hands_out_a_copier_and_cleans_up():
    rt = FakeRuntime(version=13010)
    ops = rt.ops()
    copier, why = ops.batch_copier(0)
    assert why == "" and isinstance(copier, tp.CudartBatchCopier)
    assert copier.fail_idx_form is False
    assert (copier.calls, copier.copies) == (0, 0)      # canary is not lane traffic
    kinds = [k for k, _ in rt.captured]
    assert kinds == ["flat", "3d"]
    flat_args = rt.captured[0][1]
    assert flat_args[3] >= 3                            # >= 3 flat rows
    assert rt.live == set(), "scratch must be freed"
    assert rt.streams_made == 1 and rt.streams_destroyed == 1
    assert rt.synced == [0x1001]                        # fresh stream, synchronized
    # cached per device: the canary does not run again
    assert ops.batch_copier(0) == (copier, "")
    assert len(rt.captured) == 2
    other, why2 = ops.batch_copier(1)
    assert why2 == "" and other is not copier


def test_canary_12x_runtime_uses_the_fail_idx_form():
    rt = FakeRuntime(version=12090)
    copier, why = rt.ops().batch_copier(0)
    assert why == "" and copier.fail_idx_form is True
    assert len(rt.captured[0][1]) == 9


def test_canary_misplaced_by_one_byte_is_refused_and_named():
    rt = FakeRuntime(misplace=True)
    ops = rt.ops()
    copier, why = ops.batch_copier(0)
    assert copier is None
    assert "canary" in why and "mismatch" in why
    assert rt.live == set(), "scratch freed on failure too"
    assert rt.streams_destroyed == rt.streams_made == 1
    # the refusal is cached too: no second canary
    assert ops.batch_copier(0) == (None, why)
    assert len(rt.captured) == 2


def test_canary_rc_nonzero_is_refused():
    rt = FakeRuntime(rc=801)
    copier, why = rt.ops().batch_copier(0)
    assert copier is None
    assert "canary" in why and "rc=801" in why
    assert rt.live == set()


def test_old_runtime_is_refused_by_name():
    rt = FakeRuntime(version=12070)
    copier, why = rt.ops().batch_copier(0)
    assert copier is None
    assert "12070" in why and "12080" in why and "cudaMemcpyBatchAsync" in why
    assert rt.captured == []


@pytest.mark.parametrize("sym", ["cudaMemcpyBatchAsync", "cudaMemcpy3DBatchAsync"])
def test_missing_symbol_is_refused_by_name(sym):
    rt = FakeRuntime(missing=(sym,))
    copier, why = rt.ops().batch_copier(0)
    assert copier is None and sym in why and "missing" in why


def test_any_exception_is_a_named_refusal_never_a_raise():
    rt = FakeRuntime()

    def boom(ref):
        raise OSError("driver went away")

    rt.lib.cudaRuntimeGetVersion = boom
    copier, why = rt.ops().batch_copier(0)
    assert copier is None and "OSError" in why and "driver went away" in why


def test_canary_pattern_is_non_uniform_and_the_plan_is_unaligned():
    pat = tp._canary_pattern(tp.CANARY_BYTES)
    assert len(set(pat[:512])) > 128
    assert len(tp.CANARY_FLAT) >= 3 and len(tp.CANARY_2D) >= 1
    dsts = sorted((d, d + n) for d, _s, n in tp.CANARY_FLAT)
    dsts += [(d, d + (r - 1) * dp + w) for d, _s, w, dp, _sp, r in tp.CANARY_2D]
    dsts.sort()
    for (a0, a1), (b0, b1) in zip(dsts, dsts[1:]):
        assert a1 <= b0, "canary destinations must not overlap"
    assert all(d % 16 and s % 16 for d, s, _n in tp.CANARY_FLAT)
    for _d, _s, w, dp, sp, _r in tp.CANARY_2D:
        assert dp != sp and sp != w and dp != w


# ---------------------------------------------------------------------------
# 4. DeviceOps base and desk fakes
# ---------------------------------------------------------------------------


def test_base_deviceops_names_the_absence():
    class Fake(tp.DeviceOps):
        name = "hostfake"

    assert tp.DeviceOps().batch_copier(0) == (None, "ops abstract has no batch copier")
    assert Fake().batch_copier(0) == (None, "ops hostfake has no batch copier")


def test_get_batch_copier_handles_fakes_without_the_method_and_raisers():
    class Bare:
        name = "bare"

    assert tp.get_batch_copier(Bare(), 0) == (None, "ops bare has no batch copier")

    class Raiser:
        name = "raiser"

        def batch_copier(self, device):
            raise RuntimeError("nope")

    c, why = tp.get_batch_copier(Raiser(), 0)
    assert c is None and "RuntimeError" in why and "nope" in why
    rt = FakeRuntime()
    c, why = tp.get_batch_copier(rt.ops(), 0)
    assert why == "" and c is not None


# ---------------------------------------------------------------------------
# 5. end to end through real ctypes marshaling: a gcc-built stub libcudart
# ---------------------------------------------------------------------------

STUB_C = textwrap.dedent(r"""
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>
    #include <driver_types.h>   /* structs/enums only: cuda_runtime_api.h would
                                   clash with the 12.x-form prototypes below */

    /* This stub is NOT libcudart.  It exports the entry points
       CudartDeviceOps binds, backed by host memory, and the two batch calls
       with the API's own argument checks (cudaErrorInvalidValue = 1). The
       batch functions are declared under private C names with asm labels so
       the 12.x form (extra failIdx) can differ from the header's prototype. */
    static int g_version = 13000;
    static int g_rc = 0;
    static int g_misplace = 0;
    static int g_calls = 0;
    void stub_set(int version, int rc, int misplace) {
        g_version = version; g_rc = rc; g_misplace = misplace; g_calls = 0; }
    int stub_calls(void) { return g_calls; }

    const char* cudaGetErrorString(int rc) { return "stub error"; }
    int cudaGetLastError(void) { return 0; }
    int cudaSetDevice(int d) { return 0; }
    int cudaRuntimeGetVersion(int* v) { *v = g_version; return 0; }
    int cudaStreamCreate(void** s) { *s = (void*)0x1234; return 0; }
    int cudaStreamGetFlags(void* s, unsigned* f) { *f = 0; return 0; }
    int cudaStreamDestroy(void* s) { return 0; }
    int cudaStreamSynchronize(void* s) { return 0; }
    int cudaStreamQuery(void* s) { return 0; }
    int cudaMalloc(void** p, size_t n) { *p = malloc(n); return *p ? 0 : 2; }
    int cudaFree(void* p) { free(p); return 0; }
    int cudaMemcpy(void* d, const void* s, size_t n, int k) { memcpy(d, s, n); return 0; }
    int cudaMemcpyAsync(void* d, const void* s, size_t n, int k, void* st) { memcpy(d, s, n); return 0; }
    int cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, int k, void* st) { return 0; }
    int cudaMemsetAsync(void* d, int v, size_t n, void* st) { return 0; }
    int cudaHostRegister(void* p, size_t n, unsigned f) { return 0; }
    int cudaHostUnregister(void* p) { return 0; }
    int cudaIpcGetMemHandle(void* h, void* p) { return 0; }
    int cudaIpcOpenMemHandle(void** p, cudaIpcMemHandle_t h, unsigned f) { return 0; }
    int cudaIpcCloseMemHandle(void* p) { return 0; }

    static int do_flat(void *const *dsts, const void *const *srcs, const size_t *sizes,
                       size_t count, struct cudaMemcpyAttributes *attrs, size_t *idxs,
                       size_t nattrs, cudaStream_t stream) {
        g_calls++;
        if (g_rc) return g_rc;
        if (!stream || nattrs != 1 || idxs[0] != 0) return 1;
        if (attrs[0].srcAccessOrder != cudaMemcpySrcAccessOrderStream) return 1;
        for (size_t i = 0; i < count; i++) {
            char* d = (char*)dsts[i] + ((g_misplace && i == 1) ? 1 : 0);
            memcpy(d, srcs[i], sizes[i]);
        }
        return 0;
    }
    static int do_3d(size_t n, struct cudaMemcpy3DBatchOp *ops, unsigned long long flags,
                     cudaStream_t stream) {
        g_calls++;
        if (g_rc) return g_rc;
        if (!stream || flags != 0) return 1;
        for (size_t k = 0; k < n; k++) {
            struct cudaMemcpy3DBatchOp *o = &ops[k];
            if (o->src.type != cudaMemcpyOperandTypePointer ||
                o->dst.type != cudaMemcpyOperandTypePointer) return 1;
            if (!o->extent.width || !o->extent.height || !o->extent.depth) return 1;
            if (o->srcAccessOrder != cudaMemcpySrcAccessOrderStream) return 1;
            size_t w = o->extent.width, h = o->extent.height;
            size_t dp = o->dst.op.ptr.rowLength ? o->dst.op.ptr.rowLength : w;
            size_t sp = o->src.op.ptr.rowLength ? o->src.op.ptr.rowLength : w;
            if (dp < w || sp < w || o->extent.depth != 1) return 1;
            for (size_t r = 0; r < h; r++)
                memcpy((char*)o->dst.op.ptr.ptr + r * dp,
                       (char*)o->src.op.ptr.ptr + r * sp, w);
        }
        return 0;
    }
    #ifdef FAILIDX
    int stub_batch(void *const *a, const void *const *b, const size_t *c, size_t d,
                   struct cudaMemcpyAttributes *e, size_t *f, size_t g, size_t *fail,
                   cudaStream_t s) __asm__("cudaMemcpyBatchAsync");
    int stub_batch(void *const *a, const void *const *b, const size_t *c, size_t d,
                   struct cudaMemcpyAttributes *e, size_t *f, size_t g, size_t *fail,
                   cudaStream_t s) { if (!fail) return 1; return do_flat(a, b, c, d, e, f, g, s); }
    int stub_batch3(size_t n, struct cudaMemcpy3DBatchOp *o, size_t *fail,
                    unsigned long long fl, cudaStream_t s) __asm__("cudaMemcpy3DBatchAsync");
    int stub_batch3(size_t n, struct cudaMemcpy3DBatchOp *o, size_t *fail,
                    unsigned long long fl, cudaStream_t s) { if (!fail) return 1; return do_3d(n, o, fl, s); }
    #else
    int stub_batch(void *const *a, const void *const *b, const size_t *c, size_t d,
                   struct cudaMemcpyAttributes *e, size_t *f, size_t g,
                   cudaStream_t s) __asm__("cudaMemcpyBatchAsync");
    int stub_batch(void *const *a, const void *const *b, const size_t *c, size_t d,
                   struct cudaMemcpyAttributes *e, size_t *f, size_t g,
                   cudaStream_t s) { return do_flat(a, b, c, d, e, f, g, s); }
    int stub_batch3(size_t n, struct cudaMemcpy3DBatchOp *o, unsigned long long fl,
                    cudaStream_t s) __asm__("cudaMemcpy3DBatchAsync");
    int stub_batch3(size_t n, struct cudaMemcpy3DBatchOp *o, unsigned long long fl,
                    cudaStream_t s) { return do_3d(n, o, fl, s); }
    #endif
""")


def _build_stub(tmp_path, failidx: bool):
    reason = _gcc_skip_reason()
    if reason:
        pytest.skip(reason)
    src = tmp_path / ("stub12.c" if failidx else "stub13.c")
    src.write_text(STUB_C)
    so = tmp_path / ("libstub12.so" if failidx else "libstub13.so")
    cmd = [GCC, "-shared", "-fPIC", f"-I{CU13_INCLUDE}", str(src), "-o", str(so)]
    if failidx:
        cmd.insert(1, "-DFAILIDX")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"stub does not compile (named reason): {proc.stderr[-600:]}")
    return str(so)


@pytest.mark.parametrize("failidx, version", [(False, 13000), (True, 12080)])
def test_end_to_end_real_ctypes_against_header_built_stub(tmp_path, failidx, version):
    so = _build_stub(tmp_path, failidx)
    ops = tp.CudartDeviceOps(so)
    stub = ctypes.CDLL(so)
    stub.stub_set.argtypes = [ctypes.c_int] * 3
    stub.stub_set(version, 0, 0)
    copier, why = ops.batch_copier(0)
    assert why == "" and copier is not None, why
    assert copier.fail_idx_form is failidx
    # a real batch through the real argtypes, both parts, bytes compared
    src, dst = _host(8192), ctypes.create_string_buffer(8192)
    before = stub.stub_calls()
    prep = copier.prepare(_rows_for(src, dst))
    copier.issue(prep, 0x42)
    assert stub.stub_calls() - before == 2
    raw = dst.raw
    for i in range(100):
        assert raw[3 + i] == src.raw[5 + i]
    for i in range(33):
        assert raw[500 + i] == src.raw[700 + i]
    for r in range(5):
        assert raw[2000 + r * 64:2000 + r * 64 + 40] == \
            src.raw[1000 + r * 90:1000 + r * 90 + 40]
    # untouched bytes stay untouched
    assert raw[0:3] == b"\0\0\0" and raw[3 + 100:500] == bytes(500 - 103)
    # the API-level refusal of the NULL stream is ours first
    with pytest.raises(ValueError):
        copier.issue(prep, 0)


def test_end_to_end_stub_misplacing_a_copy_is_caught_by_the_canary(tmp_path):
    so = _build_stub(tmp_path, False)
    ops = tp.CudartDeviceOps(so)
    ctypes.CDLL(so).stub_set(13000, 0, 1)
    copier, why = ops.batch_copier(0)
    assert copier is None and "canary" in why and "mismatch" in why


def test_end_to_end_stub_rc_error_is_refused_and_old_runtime_too(tmp_path):
    so = _build_stub(tmp_path, False)
    stub = ctypes.CDLL(so)
    stub.stub_set(13000, 700, 0)
    copier, why = tp.CudartDeviceOps(so).batch_copier(0)
    assert copier is None and "canary" in why and "rc=700" in why
    # the verdict is cached per process (library, device): a second runtime
    # behind the SAME path is a new process in real life
    assert tp.CudartDeviceOps(so).batch_copier(0) == (None, why)
    tp.reset_batch_copier_cache()
    stub.stub_set(12070, 0, 0)
    copier, why = tp.CudartDeviceOps(so).batch_copier(0)
    assert copier is None and "12070" in why


# ---------------------------------------------------------------------------
# review barlink-lanes2-1023: the verdict outlives the ops instance
# ---------------------------------------------------------------------------


def test_verdict_is_per_process_not_per_ops_instance():
    """The wake leg builds a fresh CudartDeviceOps per tag collect; the canary
    must run once per process and card, not once per tag."""
    rt = FakeRuntime(version=13010)
    c1, why = rt.ops().batch_copier(0)
    assert why == "" and c1 is not None
    n = len(rt.captured)
    c2, why2 = rt.ops().batch_copier(0)          # a NEW ops instance
    assert (c2, why2) == (c1, "") and len(rt.captured) == n


def test_a_canary_that_could_not_allocate_is_asked_again():
    rt = FakeRuntime(version=13010)
    real = rt.lib.cudaMalloc
    calls = {"n": 0}

    def flaky(out, nbytes):
        calls["n"] += 1
        if calls["n"] == 1:
            return 2                              # cudaErrorMemoryAllocation
        return real(out, nbytes)

    rt.lib.cudaMalloc = flaky
    c1, why = rt.ops().batch_copier(0)
    assert c1 is None and "(stage=alloc)" in why
    c2, why2 = rt.ops().batch_copier(0)
    assert why2 == "" and c2 is not None          # not cached: the next tag got it


def test_a_failing_batch_call_is_a_cached_verdict():
    rt = FakeRuntime(rc=801)
    c1, why = rt.ops().batch_copier(0)
    assert c1 is None and "(stage=batch)" in why
    n = len(rt.captured)
    assert rt.ops().batch_copier(0) == (None, why)
    assert len(rt.captured) == n
