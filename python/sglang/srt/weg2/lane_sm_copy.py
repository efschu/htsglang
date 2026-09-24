"""SM copy for the BAR1 deposit lanes (fnFL2 H22).

WHY THIS EXISTS -- the two 5090 deposit lanes of a P->D flip ran SERIALLY.
A deposit lane writes into the peer card's window through the window's BAR1
pages, which this card knows as REGISTERED HOST MEMORY
(``cudaHostRegister(..., cudaHostRegisterIoMemory)`` + ``cudaHostGetDevicePointer``,
``barlink_bar1._Cuda.register_io``/``dev_ptr``). ``cudaMemcpyAsync(...,
cudaMemcpyDefault)`` therefore classifies every deposit copy as DEVICE TO
HOST, and a GeForce card has exactly ONE copy engine per direction
(``cudaDevAttrAsyncEngineCount`` = 2 = one H2D + one D2H, measured 24.09. on
all three cards). Two lanes of one card = two streams of D2H copies = ONE
engine, which runs one copy at a time at the rate of that copy's own link:

    fnFL2x132 PP0 (746567010c, H16 lane clock), 471.7 / 479.3 MB per tag:
      p0 alone (3080 x4)  60-64 ms sync  -> 7.0-7.7 GB/s
      p1 alone (3080 x8)  30-33 ms sync  -> 13.7-15 GB/s
      together            104-110 ms each (credit 1-5, issue 2-5,
                          sync_cpu == sync: spinning on a running DMA)
    time-multiplexed engine: 471.7/7.0 + 479.3/13.7 = 102 ms  (matches)
    shared-bandwidth cap:    >= 13.7 GB/s, p0 capped at 7 -> ~70 ms (refuted)

The lever is to take the copy off the copy engine: a kernel whose threads
load from local VRAM and STORE into the mapped window -- posted PCIe writes
issued by the SMs, one kernel per lane on the lane's own stream, so the two
links fill in parallel. This is the store form the barlink collectives
already use into the same kind of mapping (``barlink_bar1_ext.writeV4``,
``st.global.wt.v4.u32``; peer pointers from the same ``register_io`` +
``dev_ptr`` pair, ``barlink_bar1.py`` ~3062/3106).

Built with NVRTC at boot (``Bar1Lanes.setup``, the lane helper thread), never
in a flip: a compile inside the scheduler thread is exactly the stall class
of memory ``jit-kernel-neue-elementgroesse-scheduler-thread``. Launched through
the driver API (``cuLaunchKernel`` via ctypes, which drops the GIL). The
kernel uses no local memory (asserted by the desk test on the PTX), so it
launches under the sleep's lowered stack limit (H15) as well.

Nothing here decides anything about slots or credits: the caller issues a
batch's pieces through :meth:`SmCopier.copy_async` / :meth:`copy2d_async`
INSTEAD OF ``ops.memcpy_async`` / ``memcpy2d_async`` on the same stream, and
the ring's ``synchronize`` of that stream still precedes the batch's ``full``
credit -- a kernel on a stream completes before a later sync of that stream
returns, exactly like the copy it replaces.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Tuple

THREADS = 256
UNROLL = 4
DEFAULT_BLOCKS = 64
KERNEL_NAME = b"weg2_lane_copy"

KERNEL_SRC = r"""
// fnFL2 H22: one lane's deposit copy, SM stores into the peer's BAR1 window.
// vec == 16: every pointer, pitch and the width are multiples of 16 bytes;
// vec == 4 : multiples of 4; vec == 1: bytes. height == 1 is a FLAT piece
// (grid-stride over the elements, 4 in flight per thread); height > 1 is a
// 2-D piece (grid-stride over the rows, the block's threads across a row).
// No 64-bit division anywhere: its helper call costs a stack frame (sm_86),
// and the kernel must launch under the sleep's lowered stack limit (H15).
#define H22_ST16(P, V)                                                          \
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"                     \
                 :: "l"(P), "r"((V).x), "r"((V).y), "r"((V).z), "r"((V).w)      \
                 : "memory")

extern "C" __global__ void __launch_bounds__(256)
weg2_lane_copy(unsigned char* __restrict__ dst, unsigned long long dpitch,
               const unsigned char* __restrict__ src, unsigned long long spitch,
               unsigned long long width, unsigned long long height, int vec)
{
    if (height == 1ull) {
        const unsigned long long nth = (unsigned long long)gridDim.x * blockDim.x;
        const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (vec == 16) {
            const unsigned long long n = width >> 4;
            const uint4* s = (const uint4*)src;
            uint4* d = (uint4*)dst;
            for (unsigned long long i = tid; i < n; i += nth * 4ull) {
                const unsigned long long i1 = i + nth, i2 = i1 + nth, i3 = i2 + nth;
                uint4 v0 = s[i], v1, v2, v3;
                if (i1 < n) v1 = s[i1];
                if (i2 < n) v2 = s[i2];
                if (i3 < n) v3 = s[i3];
                H22_ST16(d + i, v0);
                if (i1 < n) H22_ST16(d + i1, v1);
                if (i2 < n) H22_ST16(d + i2, v2);
                if (i3 < n) H22_ST16(d + i3, v3);
            }
        } else if (vec == 4) {
            const unsigned long long n = width >> 2;
            const unsigned int* s = (const unsigned int*)src;
            volatile unsigned int* d = (volatile unsigned int*)dst;
            for (unsigned long long i = tid; i < n; i += nth) d[i] = s[i];
        } else {
            volatile unsigned char* d = (volatile unsigned char*)dst;
            for (unsigned long long i = tid; i < width; i += nth) d[i] = src[i];
        }
    } else {
        for (unsigned long long r = blockIdx.x; r < height; r += gridDim.x) {
            const unsigned char* srow = src + r * spitch;
            unsigned char* drow = dst + r * dpitch;
            if (vec == 16) {
                const unsigned long long n = width >> 4;
                for (unsigned long long c = threadIdx.x; c < n; c += blockDim.x) {
                    const uint4 v = ((const uint4*)srow)[c];
                    H22_ST16(((uint4*)drow) + c, v);
                }
            } else if (vec == 4) {
                const unsigned long long n = width >> 2;
                for (unsigned long long c = threadIdx.x; c < n; c += blockDim.x)
                    ((volatile unsigned int*)drow)[c] = ((const unsigned int*)srow)[c];
            } else {
                for (unsigned long long c = threadIdx.x; c < width; c += blockDim.x)
                    ((volatile unsigned char*)drow)[c] = srow[c];
            }
        }
    }
    // the stores are posted PCIe writes into another card; fence them before
    // the kernel's completion can be observed by the host's stream sync
    __threadfence_system();
}
"""


# -- pure decisions (desk-testable) ------------------------------------------

def pick_vec(dst: int, src: int, width: int, height: int = 1,
             dpitch: int = 0, spitch: int = 0) -> int:
    """16, 4 or 1: the widest element every address of the copy is aligned to.
    A FLAT piece (height 1) ignores the pitches."""
    vals = [int(dst), int(src), int(width)]
    if int(height) > 1:
        vals += [int(dpitch), int(spitch)]
    for v in (16, 4):
        if all(x % v == 0 for x in vals):
            return v
    return 1


def split_flat(dst: int, src: int, nbytes: int):
    """A FLAT copy as (dst, src, n, vec) launches: a 16-aligned body and a
    byte tail, so a length that is not a multiple of 16 does not drop the
    whole piece to the byte loop."""
    n = int(nbytes)
    if n <= 0:
        return []
    if int(dst) % 16 == 0 and int(src) % 16 == 0 and n >= 16:
        body = n & ~15
        out = [(int(dst), int(src), body, 16)]
        if n - body:
            tail = n - body
            out.append((int(dst) + body, int(src) + body, tail, pick_vec(int(dst) + body, int(src) + body, tail)))
        return out
    return [(int(dst), int(src), n, pick_vec(dst, src, n))]


def grid_for(width: int, height: int, vec: int, blocks_cap: int) -> int:
    """Blocks for one launch, never more than the cap (the lane shares the
    card's SMs with its sibling lane). FLAT: enough for UNROLL 16-B items per
    thread. 2-D: a block per row (the kernel strides the rows over blocks)."""
    cap = int(max(1, blocks_cap))
    if int(height) > 1:
        return int(max(1, min(cap, int(height))))
    items = int(width) // max(1, int(vec))
    per_block = THREADS * (UNROLL if int(vec) == 16 else 1)
    need = (items + per_block - 1) // per_block
    return int(max(1, min(cap, need)))


def find_libnvrtc(venv: str = "") -> str:
    """The NVRTC this rig ships, cu13 first (the serving venv), then cu12."""
    root = venv or os.environ.get("VIRTUAL_ENV", "") or "/spinning/htsglang-gpu/.venv"
    site = os.path.join(root, "lib/python3.12/site-packages/nvidia")
    candidates = [
        os.path.join(site, "cu13/lib/libnvrtc.so.13"),
        os.path.join(site, "cuda_nvrtc/lib/libnvrtc.so.12"),
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "/usr/local/cuda/lib64/libnvrtc.so",
    ]
    errs = []
    for path in candidates:
        if os.sep in path and not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path)
        except OSError as exc:
            errs.append(f"{path}: {exc}")
            continue
        return path
    raise RuntimeError("no libnvrtc could be loaded (" + "; ".join(errs or candidates) + ")")


def _preload_builtins(lib_path: str) -> None:
    """NVRTC dlopens ``libnvrtc-builtins.so.<ver>`` by soname; a process whose
    LD_LIBRARY_PATH misses the venv dir (a re-exec'd worker, the desk) fails
    the compile with BUILTIN_OPERATION_FAILURE. Loaded once from the same
    directory with RTLD_GLOBAL, the soname lookup finds it."""
    d = os.path.dirname(lib_path)
    if not d:
        return
    try:
        names = sorted(n for n in os.listdir(d)
                       if n.startswith("libnvrtc-builtins.so.") and ".alt." not in n)
    except OSError:
        return
    for n in names:
        try:
            ctypes.CDLL(os.path.join(d, n), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def compile_kernel(arch: Tuple[int, int], lib_path: str = "") -> Tuple[bytes, str]:
    """NVRTC: CUBIN for sm_<arch>, else PTX for compute_<arch>. Returns
    (image, kind) with kind 'cubin' or 'ptx'. Needs no GPU."""
    lib_path = lib_path or find_libnvrtc()
    _preload_builtins(lib_path)
    lib = ctypes.CDLL(lib_path)
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    prog = ctypes.c_void_p()
    src = KERNEL_SRC.encode()

    def chk(rc, what):
        if int(rc) != 0:
            raise RuntimeError(f"{what}: {lib.nvrtcGetErrorString(int(rc)).decode()}")

    maj, mnr = int(arch[0]), int(arch[1])
    last = ""
    for kind, opt in (("cubin", f"--gpu-architecture=sm_{maj}{mnr}"),
                      ("ptx", f"--gpu-architecture=compute_{maj}{mnr}")):
        chk(lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"weg2_lane_copy.cu", 0, None, None),
            "nvrtcCreateProgram")
        try:
            opts = (ctypes.c_char_p * 2)(opt.encode(), b"-lineinfo")
            rc = lib.nvrtcCompileProgram(prog, 2, opts)
            if int(rc) != 0:
                size = ctypes.c_size_t(0)
                lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
                buf = ctypes.create_string_buffer(max(1, size.value))
                lib.nvrtcGetProgramLog(prog, buf)
                last = f"{opt}: {lib.nvrtcGetErrorString(int(rc)).decode()} {buf.value.decode(errors='replace')[:400]}"
                continue
            size = ctypes.c_size_t(0)
            if kind == "cubin":
                chk(lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "nvrtcGetCUBINSize")
                buf = ctypes.create_string_buffer(size.value)
                chk(lib.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
                return buf.raw, kind
            chk(lib.nvrtcGetPTXSize(prog, ctypes.byref(size)), "nvrtcGetPTXSize")
            buf = ctypes.create_string_buffer(size.value)
            chk(lib.nvrtcGetPTX(prog, buf), "nvrtcGetPTX")
            return buf.raw, kind
        finally:
            lib.nvrtcDestroyProgram(ctypes.byref(prog))
    raise RuntimeError(f"NVRTC refused the lane copy kernel: {last}")


# -- the launcher (driver API) ------------------------------------------------

_CU_ATTR_CC_MAJOR = 75
_CU_ATTR_CC_MINOR = 76


class SmCopier:
    """One device's loaded kernel. ``copy_async``/``copy2d_async`` take the
    same arguments as ``DeviceOps.memcpy_async``/``memcpy2d_async`` and
    enqueue on ``stream``; errors raise RuntimeError (the lane turns them
    into its named refusal)."""

    name = "sm"

    def __init__(self, device: int, *, log=None):
        self.device = int(device)
        self.blocks = DEFAULT_BLOCKS
        self.slow_bytes = 0          # bytes that could not take the 16-B path
        self.launches = 0
        self._lock = threading.Lock()
        drv = ctypes.CDLL("libcuda.so.1")
        drv.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._drv = drv
        t0 = time.perf_counter()
        self._d("cuInit", ctypes.c_uint(0))
        dev = ctypes.c_int(0)
        self._d("cuDeviceGet", ctypes.byref(dev), ctypes.c_int(self.device))
        cc = [ctypes.c_int(0), ctypes.c_int(0)]
        self._d("cuDeviceGetAttribute", ctypes.byref(cc[0]), ctypes.c_int(_CU_ATTR_CC_MAJOR), dev)
        self._d("cuDeviceGetAttribute", ctypes.byref(cc[1]), ctypes.c_int(_CU_ATTR_CC_MINOR), dev)
        self.arch = (int(cc[0].value), int(cc[1].value))
        ctx = ctypes.c_void_p()
        self._d("cuCtxGetCurrent", ctypes.byref(ctx))
        if not ctx.value:
            self._d("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
            self._d("cuCtxSetCurrent", ctx)
        self._ctx = int(ctx.value)
        image, self.image_kind = compile_kernel(self.arch)
        self.build_ms = (time.perf_counter() - t0) * 1000.0
        mod = ctypes.c_void_p()
        self._image = ctypes.create_string_buffer(image, len(image))   # kept alive
        self._d("cuModuleLoadData", ctypes.byref(mod), self._image)
        fn = ctypes.c_void_p()
        self._d("cuModuleGetFunction", ctypes.byref(fn), mod, KERNEL_NAME)
        self._mod, self._fn = mod, fn

    def _d(self, name: str, *args) -> None:
        rc = getattr(self._drv, name)(*args)
        if int(rc) != 0:
            txt = ctypes.c_char_p()
            try:
                self._drv.cuGetErrorString(int(rc), ctypes.byref(txt))
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"{name} -> {rc} ({(txt.value or b'?').decode()})")

    def _current(self) -> None:
        ctx = ctypes.c_void_p()
        self._d("cuCtxGetCurrent", ctypes.byref(ctx))
        if int(ctx.value or 0) != self._ctx:
            self._d("cuCtxSetCurrent", ctypes.c_void_p(self._ctx))

    def _launch(self, dst, dpitch, src, spitch, width, height, vec, stream, blocks) -> int:
        """Enqueue one launch; returns the bytes it moves off the 16-B path."""
        if int(width) <= 0 or int(height) <= 0:
            return 0
        grid = grid_for(width, height, vec, blocks if blocks else self.blocks)
        a = [ctypes.c_void_p(int(dst)), ctypes.c_uint64(int(dpitch)),
             ctypes.c_void_p(int(src)), ctypes.c_uint64(int(spitch)),
             ctypes.c_uint64(int(width)), ctypes.c_uint64(int(height)), ctypes.c_int(int(vec))]
        params = (ctypes.c_void_p * len(a))(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in a])
        self._current()
        self._d("cuLaunchKernel", self._fn,
                ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
                ctypes.c_uint(THREADS), ctypes.c_uint(1), ctypes.c_uint(1),
                ctypes.c_uint(0), ctypes.c_void_p(int(stream or 0)), params, None)
        with self._lock:
            self.launches += 1
            if int(vec) != 16:
                self.slow_bytes += int(width) * int(height)
        return 0 if int(vec) == 16 else int(width) * int(height)

    def copy_async(self, dst: int, src: int, nbytes: int, stream: int, *, blocks: int = 0) -> int:
        slow = 0
        for d, s, n, vec in split_flat(dst, src, nbytes):
            slow += self._launch(d, n, s, n, n, 1, vec, stream, blocks)
        return slow

    def copy2d_async(self, dst: int, dpitch: int, src: int, spitch: int,
                     width: int, height: int, stream: int, *, blocks: int = 0) -> int:
        if int(height) == 1:
            return self.copy_async(dst, src, width, stream, blocks=blocks)
        vec = pick_vec(dst, src, width, height, dpitch, spitch)
        return self._launch(dst, dpitch, src, spitch, width, height, vec, stream, blocks)


_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def copier_for(device: int, *, log=None):
    """(SmCopier, "") or (None, reason) for this process's device; built once
    (single flight), the reason is cached as well so a refused build is not
    retried per tag."""
    key = int(device)
    with _CACHE_LOCK:
        got = _CACHE.get(key)
        if got is None:
            try:
                got = (SmCopier(key, log=log), "")
            except Exception as exc:  # noqa: BLE001 -- the reason is the answer
                got = (None, f"{type(exc).__name__}: {exc}")
            _CACHE[key] = got
    return got
