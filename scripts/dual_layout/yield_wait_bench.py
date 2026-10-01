#!/usr/bin/env python3
"""DUAL-TP3PP3 lever (b) "yielding wait", measured BEFORE it is built into barlink.

Question: without MPS, P and D time-slice each card. A barlink spin kernel that
waits for a peer's flag is RUNNING work, so the driver keeps D's context on the
card for its whole slice while D does nothing. Does a wait that is NOT a
running kernel -- a stream memory operation (cuStreamWaitValue32, capturable
into a CUDA graph) or a host wait (not capturable: D's decode collectives live
in graphs) -- hand the slice to P, and what does it cost D's wake-up latency?

One D process per call, one card, its own context; the "peer" is a host
thread writing a mapped flag (the production flag is VRAM written over BAR1 by
the peer -- the scheduling question is the same: D's stream waits on an
external write). Modes:
  spin  1-block kernel spins on the flag (today's barlink shape)
  wait  cuStreamWaitValue32(flag >= want) + a 1-thread marker kernel
  host  host polls the flag, then launches the marker kernel
Per iteration: arm (enqueue the wait), sleep ~delay, write the flag (t_set),
poll the marker's done word (t_done). latency = t_done - t_set.
P load: --role p runs bf16 GEMMs (M 4096 x 5120 x 8192) for --seconds and
reports iterations/s -- P's share of the card under each D mode.

  yield_wait_bench.py --role d --dev 0 --mode spin --iters 2000 --out d.json
  yield_wait_bench.py --role p --dev 0 --seconds 30 --out p.json
No torch in the D role (cuda-python driver + NVRTC only).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
import time

SRC = r"""
extern "C" __global__ void spin(volatile unsigned *flag, unsigned want, volatile unsigned *done, unsigned tag) {
    if (threadIdx.x == 0) { while (*flag < want) { } *done = tag; __threadfence_system(); }
}
extern "C" __global__ void mark(volatile unsigned *done, unsigned tag) {
    if (threadIdx.x == 0) { *done = tag; __threadfence_system(); }
}
"""

MODES = ("spin", "wait", "host")


def preload_nvrtc_builtins():
    """libnvrtc.so.13 (nvidia/cu13/lib, loaded by cuda-python) dlopens
    libnvrtc-builtins.so.13.0 by bare name at compile time; that directory is on
    no loader path, so the compile failed with NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
    (window tatcwj, yieldwait_tatcwj_10011227). Load it RTLD_GLOBAL from the
    directory of the libnvrtc this process uses (soname match then succeeds).
    Returns the path loaded, or None if no candidate exists."""
    import glob
    import os
    import site

    roots = []
    try:
        import nvidia  # namespace package of the CUDA wheels

        roots += list(getattr(nvidia, "__path__", []))
    except ImportError:
        pass
    roots += [os.path.join(d, "nvidia") for d in site.getsitepackages()]
    for root in roots:
        for sub in ("cu13/lib", "cuda_nvrtc/lib"):
            hits = sorted(glob.glob(os.path.join(root, sub, "libnvrtc-builtins.so.13*")))
            hits = [h for h in hits if ".alt." not in h]
            if hits:
                ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
                return hits[0]
    return None


def compile_cubin(arch: str) -> bytes:
    """NVRTC-compile SRC for ``arch`` (e.g. 'sm_86'); no GPU needed."""
    from cuda.bindings import nvrtc

    preload_nvrtc_builtins()
    prog = _ck(nvrtc.nvrtcCreateProgram(SRC.encode(), b"yw.cu", 0, [], []))
    opts = [f"--gpu-architecture={arch}".encode()]
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(r[0]) != 0:
        n = _ck(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("nvrtc: " + log.decode(errors="replace"))
    n = _ck(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * n
    _ck(nvrtc.nvrtcGetCUBIN(prog, cubin))
    return cubin


def summarize(lat_us):
    """p50/p90/p99/max of a latency list in microseconds (pure)."""
    if not lat_us:
        return {"n": 0}
    s = sorted(lat_us)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1)))]  # noqa: E731
    return {"n": len(s), "p50_us": round(q(0.5), 1), "p90_us": round(q(0.9), 1),
            "p99_us": round(q(0.99), 1), "max_us": round(s[-1], 1), "mean_us": round(statistics.mean(s), 1)}


def _ck(res):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver error {err}")
    return res[1:] if isinstance(res, tuple) and len(res) > 2 else (res[1] if isinstance(res, tuple) and len(res) == 2 else None)


def run_d(a):
    from cuda.bindings import driver as cu

    _ck(cu.cuInit(0))
    dev = _ck(cu.cuDeviceGet(a.dev))
    ctx = _ck(cu.cuDevicePrimaryCtxRetain(dev))
    _ck(cu.cuCtxSetCurrent(ctx))
    major = _ck(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))
    minor = _ck(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))
    cubin = compile_cubin(f"sm_{major}{minor}")
    mod = _ck(cu.cuModuleLoadData(cubin))
    k_spin = _ck(cu.cuModuleGetFunction(mod, b"spin"))
    k_mark = _ck(cu.cuModuleGetFunction(mod, b"mark"))
    flags = cu.CU_MEMHOSTALLOC_DEVICEMAP | cu.CU_MEMHOSTALLOC_PORTABLE
    hflag = _ck(cu.cuMemHostAlloc(8, flags))
    hdone = _ck(cu.cuMemHostAlloc(8, flags))
    dflag = _ck(cu.cuMemHostGetDevicePointer(hflag, 0))
    ddone = _ck(cu.cuMemHostGetDevicePointer(hdone, 0))
    pf = ctypes.c_uint32.from_address(int(hflag))
    pd = ctypes.c_uint32.from_address(int(hdone))
    pf.value = 0
    pd.value = 0
    stream = _ck(cu.cuStreamCreate(cu.CUstream_flags.CU_STREAM_NON_BLOCKING))

    def launch(fn, args):
        import numpy as np

        vals = [np.array([int(x)], dtype=np.uint64 if i_ptr else np.uint32) for x, i_ptr in args]
        ptrs = np.array([v.ctypes.data for v in vals], dtype=np.uint64)
        _ck(cu.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, stream, ptrs.ctypes.data, 0))
        return vals, ptrs  # keep alive until the launch returned

    lat, rng = [], random.Random(a.seed)
    t_end = time.perf_counter() + a.seconds if a.seconds else None
    for i in range(a.iters):
        if t_end and time.perf_counter() > t_end:
            break
        want = i + 1
        keep = None
        if a.mode == "spin":
            keep = launch(k_spin, [(dflag, True), (want, False), (ddone, True), (want, False)])
        elif a.mode == "wait":
            _ck(cu.cuStreamWaitValue32(stream, dflag, want, int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ)))
            keep = launch(k_mark, [(ddone, True), (want, False)])
        time.sleep(a.delay_us * rng.uniform(0.5, 1.5) / 1e6)
        t_set = time.perf_counter()
        pf.value = want
        if a.mode == "host":
            keep = launch(k_mark, [(ddone, True), (want, False)])
        while pd.value != want:
            if time.perf_counter() - t_set > a.timeout_s:
                raise RuntimeError(f"iteration {i}: no completion within {a.timeout_s} s (mode {a.mode})")
        lat.append((time.perf_counter() - t_set) * 1e6)
        del keep
    _ck(cu.cuStreamSynchronize(stream))
    return {"role": "d", "dev": a.dev, "mode": a.mode, "delay_us": a.delay_us, **summarize(lat[a.warmup:])}


def run_p(a):
    import torch

    dev = torch.device("cuda", a.dev)
    x = torch.randn(a.gemm_m, 5120, dtype=torch.bfloat16, device=dev)
    w = torch.randn(5120, 8192, dtype=torch.bfloat16, device=dev) * 0.01
    for _ in range(3):
        x @ w
    torch.cuda.synchronize(dev)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        for _ in range(4):
            x @ w
        torch.cuda.synchronize(dev)
        n += 4
    dt = time.perf_counter() - t0
    return {"role": "p", "dev": a.dev, "gemm_m": a.gemm_m, "iters": n, "it_s": round(n / dt, 2),
            "tflops": round(n * 2 * a.gemm_m * 5120 * 8192 / dt / 1e12, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("d", "p"), required=True)
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="spin")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--delay-us", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--gemm-m", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run_d(a) if a.role == "d" else run_p(a)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
