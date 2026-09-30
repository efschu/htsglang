#!/usr/bin/env python3
"""DUAL-TP3PP3: does the union-arena VMM export (cuMemCreate POSIX_FD ->
cuMemImportFromShareableHandle, the weg2 union_arena_vmm path) work between two
processes on ONE card, with and without MPS?  Owner writes a pattern with
cuMemsetD32, peer maps the same physical pages and reads them back.  One JSON
line.  Driver API only (cuda.bindings), no torch, no JIT."""
import json
import os
import socket
import sys
import time
import multiprocessing as mp

N = 64 << 20
PAT = 0x5A17C0DE


def _chk(r, what):
    from cuda.bindings import driver as d
    err = r[0] if isinstance(r, tuple) else r
    if err != d.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")
    return r[1] if isinstance(r, tuple) and len(r) > 1 else None


def _ctx():
    from cuda.bindings import driver as d
    _chk(d.cuInit(0), "cuInit")
    dev = _chk(d.cuDeviceGet(0), "cuDeviceGet")
    ctx = _chk(d.cuDevicePrimaryCtxRetain(dev), "ctxRetain")
    _chk(d.cuCtxSetCurrent(ctx), "ctxSet")
    return d, dev


def _map(d, dev, handle, size):
    ptr = _chk(d.cuMemAddressReserve(size, 0, 0, 0), "reserve")
    _chk(d.cuMemMap(ptr, size, 0, handle, 0), "map")
    acc = d.CUmemAccessDesc()
    acc.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = 0
    acc.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _chk(d.cuMemSetAccess(ptr, size, [acc], 1), "setAccess")
    return ptr


def peer(path, q):
    try:
        d, dev = _ctx()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(200):
            try:
                s.connect(path)
                break
            except OSError:
                time.sleep(0.05)
        _msg, fds, _f, _a = socket.recv_fds(s, 1, 1)
        h = _chk(d.cuMemImportFromShareableHandle(
            fds[0], d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), "import")
        ptr = _map(d, dev, h, N)
        import array
        buf = array.array("I", [0]) * 1024
        _chk(d.cuMemcpyDtoH(buf, int(ptr) + N - 4096, 4096), "dtoh")
        q.put({"peer_ok": all(x == PAT for x in buf), "peer_first": hex(buf[0])})
    except Exception as e:  # noqa: BLE001
        q.put({"peer_ok": False, "peer_error": repr(e)})


def main():
    path = f"/tmp/vmm-mps-probe-{os.getpid()}.sock"
    ctxm = mp.get_context("spawn")
    q = ctxm.Queue()
    res = {}
    try:
        d, dev = _ctx()
        prop = d.CUmemAllocationProp()
        prop.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0
        prop.requestedHandleTypes = d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        h = _chk(d.cuMemCreate(N, prop, 0), "cuMemCreate(exportable)")
        ptr = _map(d, dev, h, N)
        _chk(d.cuMemsetD32(ptr, PAT, N // 4), "memset")
        _chk(d.cuCtxSynchronize(), "sync")
        fd = _chk(d.cuMemExportToShareableHandle(
            h, d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "export")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        p = ctxm.Process(target=peer, args=(path, q))
        p.start()
        conn, _ = srv.accept()
        socket.send_fds(conn, [b"F"], [int(fd)])
        res.update(q.get(timeout=60))
        p.join(timeout=30)
        res["owner_ok"] = True
    except Exception as e:  # noqa: BLE001
        res.update(owner_ok=False, owner_error=repr(e))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    res["mps"] = bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
