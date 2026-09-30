#!/usr/bin/env python3
"""DUAL-TP3PP3 risk 1b: does barlink BAR1 carry TWO groups per card at once
(D's TP3 all-reduce and P's collectives), with and without MPS?

One invocation = one group of N ranks (one process per card, gloo bootstrap
on its own port, its own ``BarlinkBar1Transport`` exactly as
``benchmark/bar1_graph_check.py`` builds it). Two invocations with the same
``--start-at`` run the two groups concurrently.

Per op the bytes are checked (bit-exact float32 pattern, every
``--verify-every``-th op) and the wall latency of one all-reduce (sync on
both sides) is recorded. ``--gemm-m`` adds a prefill-like GEMM between two
collectives (the P group's compute). A hang is caught by ``--timeout``.

  bar1_two_groups.py --name D --devs 0,1,2 --port 29711 --size 40960 --start-at T --dur 10 --out d.json
  bar1_two_groups.py --name P --devs 0,1,2 --port 29731 --size 1048576 --gemm-m 2048 --start-at T --dur 10 --out p.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _pattern(n, rank, rnd, dev):
    i = torch.arange(n, dtype=torch.float32, device=dev)
    return (rank + 1) * 1000.0 + rnd * 7.0 + (i % 97)


def _expected(n, world, rnd, dev):
    i = torch.arange(n, dtype=torch.float32, device=dev)
    return sum((r + 1) * 1000.0 for r in range(world)) + world * (rnd * 7.0 + (i % 97))


def _cold_triton_launch(dev, salt):
    """Compile AND load a triton kernel nobody has seen (the constant is new, so
    the cache misses): the host compiles, then the first launch runs
    cuModuleLoadData -- exactly what D's 'triton cold module load' did on metal
    (boot kw6pft 04:16:56, the load that never closed)."""
    import importlib.util
    import tempfile as _tf

    src = (
        "import triton, triton.language as tl\n"
        "@triton.jit\n"
        "def k(x_ptr, n, BLOCK: tl.constexpr):\n"
        "    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    m = o < n\n"
        f"    tl.store(x_ptr + o, tl.load(x_ptr + o, mask=m) * 1.0 + {(abs(hash(salt)) % 1000003) / 7.0!r}, mask=m)\n"
    )
    d = _tf.mkdtemp(prefix="coldjit")
    path = os.path.join(d, f"cold_{abs(hash(salt))}.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"cold_{abs(hash(salt))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.zeros(1024, device=dev)
    t0 = time.perf_counter()
    mod.k[(4,)](x, 1024, BLOCK=256)
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) * 1e3


def worker(rank, a, store):
    world = len(a.devs)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(a.port), RANK=str(rank),
                      WORLD_SIZE=str(world))
    res = {"group": a.name, "rank": rank, "ok": False, "err": "", "lat_ms": [], "bad": 0, "ops": 0}
    t = None
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        torch.cuda.set_device(a.devs[rank])
        dev = torch.device("cuda", a.devs[rank])
        torch.zeros(1, device=dev)
        from sglang.srt.distributed.device_communicators.barlink_bar1 import BarlinkBar1Transport
        from sglang.srt.distributed.device_communicators.barlink_matrix_transport import _window_bytes

        extra = []
        if a.windows_mib:
            # the production shape: several barlink groups per process, each its
            # own BAR1 window (D: world/tp/dcp, P: world/pp) -- built one after
            # the other like GroupCoordinator does, all kept alive.
            mibs = [int(x) for x in a.windows_mib.split(",")]
            t = BarlinkBar1Transport(dist.group.WORLD, dev, mibs[0] << 20)
            for m in mibs[1:]:
                extra.append(BarlinkBar1Transport(dist.group.WORLD, dev, m << 20))
        else:
            t = BarlinkBar1Transport(dist.group.WORLD, dev, _window_bytes())
        dist.barrier()
        if a.bar1_snapshot:
            import subprocess as _sp
            res["bar1_free_mib"] = _sp.run(
                ["nvidia-smi", "--query-gpu=index,name,pci.bus_id", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip()
            q = _sp.run(["nvidia-smi", "-q", "-d", "MEMORY"], capture_output=True, text=True).stdout
            res["bar1_q"] = [l.strip() for l in q.splitlines() if l.strip().startswith(("Free", "Used", "Total"))][-24:]
        res["proof"] = {str(k): bool(v) for k, v in t.byte_proof_all().items()}
        if not all(res["proof"].values()):
            raise RuntimeError(f"byte proof failed: {res['proof']}")
        n = a.size // 4
        res["handles"] = bool(t.handles("all_reduce", a.size))
        if not res["handles"]:
            raise RuntimeError(f"handles(all_reduce, {a.size}) -> False")
        gemm = None
        if a.gemm_m:
            ga = torch.randn(a.gemm_m, 5120, dtype=torch.bfloat16, device=dev)
            gb = torch.randn(5120, 8192, dtype=torch.bfloat16, device=dev) * 0.01
            gemm = lambda: ga @ gb  # noqa: E731
        for w in range(3):
            t.barlink_all_reduce(None, _pattern(n, rank, 0, dev))
        torch.cuda.synchronize(dev)
        dist.barrier()
        now = time.time()
        if a.start_at > now:
            time.sleep(a.start_at - now)
        dist.barrier()
        t_end = time.perf_counter() + a.dur
        t_jit = (time.perf_counter() + a.jit_at) if a.jit_at > 0 else None
        res["jit_ms"] = []
        rnd = 0
        while time.perf_counter() < t_end:
            rnd += 1
            if gemm is not None:
                gemm()
            if t_jit is not None and time.perf_counter() >= t_jit and (a.jit_ranks == "all" or str(rank) in a.jit_ranks.split(",")):
                # the cold load lands while this group's previous collective and the
                # OTHER group's kernels may still be resident (async launch)
                inp_j = _pattern(n, rank, rnd, dev)
                t.barlink_all_reduce(None, inp_j)  # left in flight, not synced
                res["jit_ms"].append(round(_cold_triton_launch(dev, f"{a.name}{rank}{rnd}{time.time()}"), 1))
                t_jit = (time.perf_counter() + a.jit_every) if a.jit_every > 0 else None
            if a.gemm_only:
                torch.cuda.synchronize(dev)
            else:
                inp = _pattern(n, rank, rnd, dev)
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                out = t.barlink_all_reduce(None, inp)
                torch.cuda.synchronize(dev)
                res["lat_ms"].append((time.perf_counter() - t0) * 1e3)
                if rnd % a.verify_every == 0:
                    if not torch.equal(out, _expected(n, world, rnd, dev)):
                        res["bad"] += 1
            # keep the group in lockstep on the host side (as the ranks of one
            # TP group are): the shortest loop decides the op count.
            flag = torch.tensor([1 if time.perf_counter() < t_end else 0])
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 0:
                break
        res["ops"] = rnd
        res["ok"] = res["bad"] == 0
    except BaseException as e:  # noqa: BLE001
        res["err"] = f"{type(e).__name__}: {e}"[:600]
        traceback.print_exc()
    finally:
        try:
            if t is not None:
                t.close()
            for x in locals().get("extra", []) or []:
                x.close()
        except Exception:  # noqa: BLE001
            pass
        lat = sorted(res.pop("lat_ms"))
        if lat:
            res.update(p50_ms=statistics.median(lat), p99_ms=lat[min(len(lat) - 1, int(0.99 * len(lat)))],
                       max_ms=lat[-1], n=len(lat))
        pathlib.Path(store, f"r{rank}.json").write_text(json.dumps(res))
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--devs", default="0,1,2")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--size", type=int, default=40960)
    ap.add_argument("--gemm-m", type=int, default=0)
    ap.add_argument("--start-at", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--verify-every", type=int, default=10)
    ap.add_argument("--gemm-only", action="store_true",
                    help="P's compute alone: the GEMM loop, no collectives (the transport is still built)")
    ap.add_argument("--jit-at", type=float, default=0.0,
                    help="seconds into the loop: this group loads a COLD triton module (0 = never)")
    ap.add_argument("--jit-every", type=float, default=0.0, help="repeat the cold load every N s (0 = once)")
    ap.add_argument("--jit-ranks", default="all", help="'all' or a comma list of ranks that do the cold load")
    ap.add_argument("--windows-mib", default="",
                    help="comma list: one BAR1 window per barlink group, e.g. D '16,32,40', P '16,64'")
    ap.add_argument("--bar1-snapshot", action="store_true", help="record nvidia-smi BAR1 usage once all windows exist")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.devs = [int(x) for x in a.devs.split(",")]
    with tempfile.TemporaryDirectory() as store:
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=worker, args=(r, a, store)) for r in range(len(a.devs))]
        [p.start() for p in ps]
        [p.join() for p in ps]
        ranks = []
        for r in range(len(a.devs)):
            f = pathlib.Path(store, f"r{r}.json")
            ranks.append(json.loads(f.read_text()) if f.exists() else {"rank": r, "ok": False, "err": "no result"})
    out = {"group": a.name, "size": a.size, "gemm_m": a.gemm_m,
           "mps": bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY")),
           "sm_pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
           "ok": all(x.get("ok") for x in ranks), "ranks": ranks}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("group", "ok", "mps")}),
          [(x.get("rank"), x.get("p50_ms"), x.get("p99_ms"), x.get("bad"), x.get("err", "")[:120]) for x in ranks])


if __name__ == "__main__":
    main()
