#!/usr/bin/env python3
"""#274 slice D, S2: is there an SM allocator inside ONE process, and does a
captured graph survive a change of the split?

Green contexts are the only real SM allocator available to a single-process
multi-lane runtime: MPS' ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` is read by a
client when it CREATES its context, so it is per process, and using it would
mean going back to process-lanes and losing the shared weight bytes that are
the point of the whole strand (DESIGN_201 addendum 12 (2)).

Two questions, and the second one decides more than the first:

  Q1  Do ``cuGreenCtxCreate`` and friends work on sm86 AND sm120 with this
      driver, and at what SM granularity?
  Q2  A graph is bound to the context it was captured in.  If changing the SM
      split forces a RE-CAPTURE, then the action space of any future
      controller is a small, pre-captured ladder of fixed splits and nothing
      else -- addendum 12 (4) calls re-capture forbidden, and this probe is
      what turns that from an assumption into a verdict.

Pure driver API through ctypes: no torch, no nvrtc, no kernel of our own.
The captured work is ``cuMemsetD32Async``, which is capturable and whose
result is checkable, so a graph that "replayed" without doing anything cannot
pass.

Usage::

    python3 scripts/dual_group/green_ctx_probe.py            # all devices
    python3 scripts/dual_group/green_ctx_probe.py --json out.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# driver bindings
# ---------------------------------------------------------------------------

CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 0x1
CU_STREAM_CAPTURE_MODE_GLOBAL = 0
CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING = 0x1
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16


class CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class _ResUnion(ctypes.Union):
    _fields_ = [("sm", CUdevSmResource), ("_oversize", ctypes.c_ubyte * 48)]


class CUdevResource(ctypes.Structure):
    """Layout from cuda.h: type, 92 bytes of internal padding, 48-byte union."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("_internal_padding", ctypes.c_ubyte * 92),
        ("u", _ResUnion),
    ]


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.missing: List[str] = []
        for name, argtypes in (
            ("cuInit", [ctypes.c_uint]),
            ("cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]),
            ("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]),
            ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
            ("cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
            (
                "cuDeviceGetAttribute",
                [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
            ),
            (
                "cuDevicePrimaryCtxRetain",
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            ),
            ("cuDevicePrimaryCtxRelease", [ctypes.c_int]),
            ("cuCtxSetCurrent", [ctypes.c_void_p]),
            ("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
            ("cuMemFree_v2", [ctypes.c_void_p]),
            (
                "cuMemcpyDtoH_v2",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
            ),
            ("cuMemsetD32_v2", [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t]),
            (
                "cuMemsetD32Async",
                [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p],
            ),
            ("cuStreamCreate", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]),
            ("cuStreamDestroy_v2", [ctypes.c_void_p]),
            ("cuStreamSynchronize", [ctypes.c_void_p]),
            ("cuStreamBeginCapture_v2", [ctypes.c_void_p, ctypes.c_int]),
            (
                "cuStreamEndCapture",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            ),
            (
                "cuGraphInstantiateWithFlags",
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong],
            ),
            ("cuGraphLaunch", [ctypes.c_void_p, ctypes.c_void_p]),
            ("cuGraphExecDestroy", [ctypes.c_void_p]),
            ("cuGraphDestroy", [ctypes.c_void_p]),
            (
                "cuDeviceGetDevResource",
                [ctypes.c_int, ctypes.POINTER(CUdevResource), ctypes.c_int],
            ),
            (
                "cuDevSmResourceSplitByCount",
                [
                    ctypes.POINTER(CUdevResource),
                    ctypes.POINTER(ctypes.c_uint),
                    ctypes.POINTER(CUdevResource),
                    ctypes.POINTER(CUdevResource),
                    ctypes.c_uint,
                    ctypes.c_uint,
                ],
            ),
            (
                "cuDevResourceGenerateDesc",
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(CUdevResource),
                    ctypes.c_uint,
                ],
            ),
            (
                "cuGreenCtxCreate",
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_uint,
                ],
            ),
            ("cuGreenCtxDestroy", [ctypes.c_void_p]),
            (
                "cuGreenCtxStreamCreate",
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.c_int,
                ],
            ),
            (
                "cuCtxFromGreenCtx",
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            ),
            (
                "cuGreenCtxGetDevResource",
                [ctypes.c_void_p, ctypes.POINTER(CUdevResource), ctypes.c_int],
            ),
        ):
            try:
                fn = getattr(self.lib, name)
            except AttributeError:
                self.missing.append(name)
                continue
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            setattr(self, name, fn)
        self.lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorName.restype = ctypes.c_int

    def err(self, code: int) -> str:
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(code, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
        return f"CUDA_ERROR_{code}"

    def check(self, code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} -> {self.err(code)}")


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------


def _split(cu: Cuda, total: CUdevResource, min_count: int):
    """One split attempt: returns (groups, sm_per_group, remaining_sm)."""
    nb = ctypes.c_uint(0)
    remaining = CUdevResource()
    # First call with a NULL result asks only for the group COUNT.
    rc = cu.cuDevSmResourceSplitByCount(
        None,
        ctypes.byref(nb),
        ctypes.byref(total),
        ctypes.byref(remaining),
        0,
        ctypes.c_uint(min_count),
    )
    if rc != 0:
        return None, None, None, cu.err(rc)
    n = int(nb.value)
    if n <= 0:
        return 0, None, None, ""
    groups = (CUdevResource * n)()
    nb2 = ctypes.c_uint(n)
    rc = cu.cuDevSmResourceSplitByCount(
        groups,
        ctypes.byref(nb2),
        ctypes.byref(total),
        ctypes.byref(remaining),
        0,
        ctypes.c_uint(min_count),
    )
    if rc != 0:
        return None, None, None, cu.err(rc)
    return (
        groups,
        [int(groups[i].u.sm.smCount) for i in range(int(nb2.value))],
        int(remaining.u.sm.smCount),
        "",
    )


def _green_ctx(cu: Cuda, dev: int, group: CUdevResource):
    desc = ctypes.c_void_p()
    cu.check(
        cu.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(group), 1),
        "cuDevResourceGenerateDesc",
    )
    gctx = ctypes.c_void_p()
    cu.check(
        cu.cuGreenCtxCreate(ctypes.byref(gctx), desc, dev, CU_GREEN_CTX_DEFAULT_STREAM),
        "cuGreenCtxCreate",
    )
    stream = ctypes.c_void_p()
    cu.check(
        cu.cuGreenCtxStreamCreate(ctypes.byref(stream), gctx, 0x1, 0),
        "cuGreenCtxStreamCreate",
    )
    return gctx, stream


def _time_stream(cu: Cuda, fn, reps: int = 7) -> float:
    """Median wall ms of ``fn`` (which must end in a stream synchronize)."""
    import statistics
    import time

    fn()  # warm
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(xs)


def _mask_effect(
    cu: Cuda, dev: ctypes.c_int, total: CUdevResource, sm_count: int
) -> Dict[str, Any]:
    """Wide vs narrow SM split, eager and replayed, on one big memset."""
    out: Dict[str, Any] = {}
    wide_groups, wide_sizes, _r, err_w = _split(cu, total, max(1, sm_count // 2))
    narrow_groups, narrow_sizes, _r2, err_n = _split(cu, total, 1)
    if not wide_sizes or not narrow_sizes:
        return {"ok": False, "error": err_w or err_n or "split failed"}
    gw, sw = _green_ctx(cu, dev, wide_groups[0])
    gn, sn = _green_ctx(cu, dev, narrow_groups[0])
    out["wide_sm"] = wide_sizes[0]
    out["narrow_sm"] = narrow_sizes[0]

    n = 64 * 1024 * 1024  # 256 MiB of uint32
    ptr = ctypes.c_void_p()
    rc = cu.cuMemAlloc_v2(ctypes.byref(ptr), n * 4)
    if rc != 0:
        cu.cuStreamDestroy_v2(sw)
        cu.cuStreamDestroy_v2(sn)
        cu.cuGreenCtxDestroy(gw)
        cu.cuGreenCtxDestroy(gn)
        return {"ok": False, "error": f"cuMemAlloc: {cu.err(rc)}"}

    def eager(stream):
        def go():
            for _ in range(8):
                cu.cuMemsetD32Async(ptr, 0x5A5A5A5A, n, stream)
            cu.cuStreamSynchronize(stream)

        return go

    out["eager_wide_ms"] = round(_time_stream(cu, eager(sw)), 3)
    out["eager_narrow_ms"] = round(_time_stream(cu, eager(sn)), 3)
    out["eager_ratio"] = round(
        out["eager_narrow_ms"] / max(out["eager_wide_ms"], 1e-9), 3
    )

    graph = ctypes.c_void_p()
    exec_ = ctypes.c_void_p()
    rc = cu.cuStreamBeginCapture_v2(sw, CU_STREAM_CAPTURE_MODE_GLOBAL)
    if rc == 0:
        for _ in range(8):
            cu.cuMemsetD32Async(ptr, 0x5A5A5A5A, n, sw)
        rc = cu.cuStreamEndCapture(sw, ctypes.byref(graph))
    if rc == 0:
        rc = cu.cuGraphInstantiateWithFlags(ctypes.byref(exec_), graph, 0)
    if rc != 0:
        out["ok"] = False
        out["error"] = f"capture/instantiate: {cu.err(rc)}"
    else:

        def replay(stream):
            def go():
                cu.cuGraphLaunch(exec_, stream)
                cu.cuStreamSynchronize(stream)

            return go

        out["graph_on_wide_ms"] = round(_time_stream(cu, replay(sw)), 3)
        out["graph_on_narrow_ms"] = round(_time_stream(cu, replay(sn)), 3)
        # Which eager point does the narrow replay resemble?  If it looks like
        # the WIDE one, the exec carries its capture context's mask and the
        # launching stream is decoration; if it looks like the NARROW one, the
        # stream governs and a single capture can be re-masked at runtime.
        gn_ms = out["graph_on_narrow_ms"]
        d_wide = abs(gn_ms - out["eager_wide_ms"])
        d_narrow = abs(gn_ms - out["eager_narrow_ms"])
        if out["eager_ratio"] < 1.15:
            out["governed_by"] = "undecidable"
            out["note"] = (
                "the narrow split is not measurably slower on this workload, "
                "so the comparison cannot separate the two hypotheses"
            )
        else:
            out["governed_by"] = (
                "capture context" if d_wide < d_narrow else "launching stream"
            )
        out["ok"] = True
        cu.cuGraphExecDestroy(exec_)
        cu.cuGraphDestroy(graph)

    cu.cuMemFree_v2(ptr)
    cu.cuStreamDestroy_v2(sw)
    cu.cuStreamDestroy_v2(sn)
    cu.cuGreenCtxDestroy(gw)
    cu.cuGreenCtxDestroy(gn)
    return out


def probe_device(cu: Cuda, index: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"index": index}
    dev = ctypes.c_int()
    cu.check(cu.cuDeviceGet(ctypes.byref(dev), index), "cuDeviceGet")
    name = ctypes.create_string_buffer(128)
    cu.cuDeviceGetName(name, 128, dev)
    out["name"] = name.value.decode()
    for key, attr in (
        ("cc_major", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        ("cc_minor", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        ("sm_count", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
    ):
        v = ctypes.c_int()
        cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
        out[key] = int(v.value)
    out["arch"] = f"sm{out['cc_major']}{out['cc_minor']}"

    pctx = ctypes.c_void_p()
    cu.check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), dev), "PrimaryCtxRetain")
    cu.check(cu.cuCtxSetCurrent(pctx), "cuCtxSetCurrent")

    try:
        total = CUdevResource()
        rc = cu.cuDeviceGetDevResource(
            dev, ctypes.byref(total), CU_DEV_RESOURCE_TYPE_SM
        )
        if rc != 0:
            out["q1"] = {"ok": False, "error": f"cuDeviceGetDevResource: {cu.err(rc)}"}
            return out
        out["sm_resource_total"] = int(total.u.sm.smCount)

        # --- Q1: granularity ladder -----------------------------------
        ladder = []
        for want in (1, 2, 4, 8, 16, out["sm_count"] // 4, out["sm_count"] // 2):
            if want < 1:
                continue
            groups, sizes, remaining, err = _split(cu, total, want)
            ladder.append(
                {
                    "min_count": want,
                    "groups": None if sizes is None else len(sizes),
                    "sm_per_group": sizes,
                    "remaining_sm": remaining,
                    "error": err,
                }
            )
        out["split_ladder"] = ladder
        granular = sorted(
            {s for row in ladder if row["sm_per_group"] for s in row["sm_per_group"]}
        )
        out["granularity_sm"] = granular[0] if granular else None

        half = max(1, out["sm_count"] // 2)
        groups, sizes, remaining, err = _split(cu, total, half)
        if not sizes:
            out["q1"] = {"ok": False, "error": err or "split produced no group"}
            return out
        out["q1"] = {
            "ok": True,
            "split_at": half,
            "sm_per_group": sizes,
            "remaining_sm": remaining,
        }

        # --- create two DIFFERENT green contexts ----------------------
        gctx_a, stream_a = _green_ctx(cu, dev, groups[0])
        quarter = max(1, out["sm_count"] // 4)
        groups_b, sizes_b, _rem_b, err_b = _split(cu, total, quarter)
        if not sizes_b:
            out["q2"] = {"ok": False, "error": f"second split failed: {err_b}"}
            return out
        gctx_b, stream_b = _green_ctx(cu, dev, groups_b[0])
        out["ctx_a_sm"] = sizes[0]
        out["ctx_b_sm"] = sizes_b[0]

        # --- Q2: capture in A, replay in B ----------------------------
        n = 1024
        ptr = ctypes.c_void_p()
        cu.check(cu.cuMemAlloc_v2(ctypes.byref(ptr), n * 4), "cuMemAlloc")
        cu.check(cu.cuMemsetD32_v2(ptr, 0, n), "cuMemsetD32")

        cu.check(
            cu.cuStreamBeginCapture_v2(stream_a, CU_STREAM_CAPTURE_MODE_GLOBAL),
            "cuStreamBeginCapture",
        )
        cu.check(cu.cuMemsetD32Async(ptr, 0xA5A5A5A5, n, stream_a), "memsetD32Async")
        graph = ctypes.c_void_p()
        cu.check(
            cu.cuStreamEndCapture(stream_a, ctypes.byref(graph)), "cuStreamEndCapture"
        )
        exec_ = ctypes.c_void_p()
        cu.check(
            cu.cuGraphInstantiateWithFlags(ctypes.byref(exec_), graph, 0),
            "cuGraphInstantiate",
        )

        q2: Dict[str, Any] = {"captured_in": "green ctx A"}

        def _launch(stream, tag):
            cu.check(cu.cuMemsetD32_v2(ptr, 0, n), "reset")
            rc = cu.cuGraphLaunch(exec_, stream)
            if rc != 0:
                return {"launched": False, "error": cu.err(rc)}
            rc = cu.cuStreamSynchronize(stream)
            if rc != 0:
                return {"launched": True, "synced": False, "error": cu.err(rc)}
            host = (ctypes.c_uint * 4)()
            cu.check(cu.cuMemcpyDtoH_v2(host, ptr, 16), "cuMemcpyDtoH")
            ok = all(host[i] == 0xA5A5A5A5 for i in range(4))
            return {"launched": True, "synced": True, "work_done": ok, "tag": tag}

        q2["replay_in_a"] = _launch(stream_a, "A")
        q2["replay_in_b_other_split"] = _launch(stream_b, "B")

        plain = ctypes.c_void_p()
        cu.check(cu.cuStreamCreate(ctypes.byref(plain), 0), "cuStreamCreate")
        q2["replay_on_plain_stream"] = _launch(plain, "plain")

        # The hard form of the question: destroy the context the graph was
        # captured in, then replay.  If THAT works, a controller may retire a
        # rung without re-capturing everything captured under it.
        cu.cuStreamDestroy_v2(stream_a)
        rc_destroy = cu.cuGreenCtxDestroy(gctx_a)
        q2["destroy_a"] = cu.err(rc_destroy) if rc_destroy else "ok"
        if rc_destroy == 0:
            q2["replay_after_a_destroyed"] = _launch(stream_b, "B-after-destroy")

        # --- Q3: does the LAUNCHING stream's mask govern the replay? ----
        # "It launched and did work" is not the same as "it ran under B's SM
        # split": a memset proves the graph survives, not which mask it ran
        # under, and the two answers point at different controllers.  So:
        # calibrate on eager work whether a narrow split is measurably slower
        # at all, and only then compare a WIDE-captured graph replayed on the
        # narrow stream against both eager points.
        out["q3"] = _mask_effect(cu, dev, total, out["sm_count"])

        out["q2"] = q2

        cu.cuGraphExecDestroy(exec_)
        cu.cuGraphDestroy(graph)
        cu.cuStreamDestroy_v2(plain)
        cu.cuStreamDestroy_v2(stream_b)
        cu.cuGreenCtxDestroy(gctx_b)
        cu.cuMemFree_v2(ptr)
    except RuntimeError as e:
        out.setdefault("q2", {})["error"] = str(e)
    finally:
        cu.cuDevicePrimaryCtxRelease(dev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()

    cu = Cuda()
    result: Dict[str, Any] = {"missing_symbols": cu.missing}
    cu.check(cu.cuInit(0), "cuInit")
    ver = ctypes.c_int()
    cu.cuDriverGetVersion(ctypes.byref(ver))
    result["driver_api_version"] = int(ver.value)
    cnt = ctypes.c_int()
    cu.check(cu.cuDeviceGetCount(ctypes.byref(cnt)), "cuDeviceGetCount")
    result["device_count"] = int(cnt.value)

    if cu.missing:
        result["verdict"] = "green contexts unavailable: " + ", ".join(cu.missing)
        print(json.dumps(result, indent=2))
        return 1

    devices = [args.device] if args.device is not None else list(range(int(cnt.value)))
    result["devices"] = [probe_device(cu, i) for i in devices]

    archs = {d.get("arch"): d for d in result["devices"]}
    q1 = all(d.get("q1", {}).get("ok") for d in result["devices"])
    survive: Optional[bool] = None
    rows = [d.get("q2", {}) for d in result["devices"]]
    if all("replay_in_b_other_split" in r for r in rows):
        survive = all(r["replay_in_b_other_split"].get("work_done") for r in rows)
    governed = sorted(
        {
            d.get("q3", {}).get("governed_by")
            for d in result["devices"]
            if d.get("q3", {}).get("governed_by")
        }
    )
    result["verdict"] = {
        "archs": sorted(a for a in archs if a),
        "green_ctx_creatable": q1,
        "granularity_sm": {
            d.get("arch"): d.get("granularity_sm") for d in result["devices"]
        },
        "graph_survives_split_change": survive,
        "graph_replay_governed_by": governed,
        "graph_survives_capture_ctx_destroy": all(
            d.get("q2", {}).get("replay_after_a_destroyed", {}).get("launched") is False
            for d in result["devices"]
        )
        is False,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
