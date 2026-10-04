#!/usr/bin/env python3
"""METAL TEST of the P green-context ladder (item 1330). PREPARED, NOT RUN (needs a gpuq window on ONE card).

Run inside the image / venv on a card whose window you hold, as an MPS client with the percentage UNSET
(``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` must not be set -- the ladder and a static percentage exclude each other):

    CUDA_VISIBLE_DEVICES=<one card> python3 scripts/dual_group/green_ladder_metal_1330.py --out /spinning/gpu-arb/probe_out/green_ladder_1330/<card>.json

It drives the REAL ``weg2/dual_green.py`` (CtypesBackend, GreenLadder with its probe, GreenActuator.pick with the
hook's wait_stream pattern of ``_pp_launch_batch``) with a stand-in load (a P-like chunk of bf16 GEMMs, a D-like
graph of dependent small GEMMs). What it answers (each is one line ``G1330 <key> ...`` and a JSON field):

  W1  the ladder as the driver built it: SM per stage, the boot probe's per-stage time ratio and effect.
  W2  stage walk 100 -> 75 -> 50 -> 25 -> 50 -> 75 -> 100 (x3) through ``pick``: the chunk time of every stage
      (the P-side cost of a rung) and that every ``pick`` handed back the stream of the wanted stage.
  W3  NEEDLE TEST per stage (hang test): ``--chunks`` chunks per stage under a per-chunk watchdog
      (``--hang-s``); a chunk that does not finish prints ``G1330 HANG stage=<k>`` and exits 3 -- the operator
      kills the process, nothing here retries. This is the 'narrow mask hangs a kernel that assumes co-residence'
      class (spin over more CTAs than mask SMs, split-K semaphores).
  W4  switch cost: alternating two stages every chunk against the same-stage control (host microseconds of the
      pick + the extra wait_stream pair, and the chunk-time delta).
  W5  the D stand-in's round (a captured graph on the primary stream) per P stage while P chunks run on the
      stage's stream: the single number the ladder exists for (compare with the 100 % row).
  W6  VRAM: the context cost of the ladder (``mem_get_info`` before the build / after the build / after the walk).

NOT covered here (needs the real boot, see the report): the prefill graph going eager per stage, the P
allocator pools per green stream with real activations, PP wire uniformity across the three stages, the
Barlink collectives next to a narrow stream, D rounds with the real D.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--chunks", type=int, default=40, help="chunks per stage for the needle test")
    ap.add_argument("--hang-s", type=float, default=20.0, help="per-chunk watchdog")
    ap.add_argument("--m", type=int, default=1024, help="P-like chunk tokens")
    ap.add_argument("--n-gemm", type=int, default=64, help="GEMM launches per P chunk")
    args = ap.parse_args()
    if os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
        print("G1330 REFUSED CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set: the ladder cannot lift a static percentage")
        return 2

    import torch

    from sglang.srt.weg2 import dual_green as G
    from sglang.srt.weg2 import dual_share as S

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    out: dict = {"card": torch.cuda.get_device_name(0), "mps_pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}
    free0 = torch.cuda.mem_get_info()[0]

    # ---- W1: build + probe -------------------------------------------------------------------------
    logs, warns = [], []
    be = G.CtypesBackend(0)
    ladder = G.GreenLadder(be, S.ShareConfig().rungs, probe=True, log=logs.append, warn=warns.append).build()
    print("G1330 W1", ladder.marker())
    for w in warns:
        print("G1330 W1 WARN", w)
    out["w1"] = {"marker": ladder.marker(), "warns": warns,
                 "rungs": {i: {"f": r.fraction, "sm": r.sm, "time_ratio": r.timed_ratio, "effect": r.effect}
                           for i, r in ladder.rungs.items()}, "failed": ladder.failed}
    free1 = torch.cuda.mem_get_info()[0]
    out["w6_build_mib"] = (free0 - free1) / 2**20
    print(f"G1330 W6 ladder build cost {(free0 - free1) / 2**20:.1f} MiB (contexts+streams, probe tensors freed)")
    if ladder.healthy_count == 0:
        print("G1330 NO-RUNG the ladder serves nothing on this card/driver: the fallback actuators would take over")
        _dump(args.out, out)
        return 0

    # ---- the real actuator with real streams ----------------------------------------------------------
    fwd, sch = torch.cuda.Stream(), torch.cuda.Stream()
    sched = SimpleNamespace(forward_stream=fwd, schedule_stream=sch, forward_stream_ctx=torch.cuda.stream(fwd),
                            device_module=torch.cuda)

    class R:
        def read(self):
            return S.CtlState()

    act = G.GreenActuator(ladder, R(), pp_rank=1, pp_size=3, log=lambda m: print("G1330 act", m))
    fractions = ladder.fractions
    a = torch.randn(args.m, 5120, dtype=torch.bfloat16, device=dev)
    b = torch.randn(5120, 5120, dtype=torch.bfloat16, device=dev)
    c = torch.empty(args.m, 5120, dtype=torch.bfloat16, device=dev)

    def chunk():
        for _ in range(args.n_gemm):
            torch.matmul(a, b, out=c)

    def run_stage(rung: int, seq: int, watchdog: bool = True):
        """The hook of _pp_launch_batch, verbatim in its stream calls; returns (chunk ms, host pick us, stream)."""
        act.apply(G.Weg2DualGreenRung(seq, rung, int(round(fractions[rung] * 1e6))))
        t0 = time.perf_counter()
        ctx, gc = act.pick(sched)
        pick_us = (time.perf_counter() - t0) * 1e6
        with ctx:
            sched.forward_stream.wait_stream(sched.schedule_stream)
            if gc is not None:
                gc.wait_stream(sched.forward_stream)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            chunk()
            e1.record()
            done = torch.cuda.Event()
            done.record()
            if gc is not None:
                sched.forward_stream.wait_stream(gc)
        t_start = time.perf_counter()
        while not done.query():
            if watchdog and time.perf_counter() - t_start > args.hang_s:
                print(f"G1330 HANG stage={rung} f={fractions[rung]:.2f} sm={act.active_sm} after {args.hang_s}s")
                sys.stdout.flush()
                os._exit(3)
            time.sleep(0.0005)
        torch.cuda.synchronize()
        return e0.elapsed_time(e1), pick_us, gc

    seq = 0
    served = [i for i in range(len(fractions)) if i == 0 or i in ladder.rungs]

    # ---- W2: the stage walk ------------------------------------------------------------------------------
    walk = served + served[-2::-1] + served[1:] + served[-2::-1]
    per_stage: dict = {i: [] for i in served}
    wrong = 0
    for rung in walk:
        for _ in range(3):
            seq += 1
            ms, pick_us, gc = run_stage(rung, seq)
            per_stage[rung].append(ms)
            want_stream = None if rung == 0 else ladder.rungs[rung].stream
            if gc is not want_stream:
                wrong += 1
    base = statistics.median(per_stage[0]) if per_stage.get(0) else None
    w2 = {}
    for i in served:
        med = statistics.median(per_stage[i])
        w2[i] = {"f": fractions[i], "sm": (ladder.rungs[i].sm if i else None), "chunk_ms_p50": med,
                 "x_vs_100": (med / base if base else None)}
        print(f"G1330 W2 stage={i} f={fractions[i]:.2f} sm={w2[i]['sm']} chunk_ms_p50={med:.2f} x_vs_100={w2[i]['x_vs_100']:.2f}")
    print(f"G1330 W2 wrong_stream_picks={wrong} (must be 0)")
    out["w2"] = w2
    out["w2_wrong_stream_picks"] = wrong

    # ---- W3: needle (hang) test per stage ---------------------------------------------------------------------
    for i in served:
        t0 = time.perf_counter()
        for _ in range(args.chunks):
            seq += 1
            run_stage(i, seq)
        print(f"G1330 W3 stage={i} f={fractions[i]:.2f} {args.chunks} chunks without a hang in {time.perf_counter() - t0:.1f}s")
    out["w3"] = "no hang on any served stage"

    # ---- W4: switch cost ----------------------------------------------------------------------------------------
    sw = {}
    for lo, hi in [(served[0], served[1])] + ([(served[1], served[-1])] if len(served) > 2 else []):
        ctl_ms, alt_ms, alt_pick = [], [], []
        for k in range(60):
            seq += 1
            ms, _p, _g = run_stage(lo, seq)
            ctl_ms.append(ms)
        for k in range(60):
            seq += 1
            ms, p, _g = run_stage(lo if k % 2 == 0 else hi, seq)
            if k % 2 == 0:
                alt_ms.append(ms)
                alt_pick.append(p)
        d = statistics.median(alt_ms) - statistics.median(ctl_ms)
        sw[f"{lo}<->{hi}"] = {"chunk_delta_ms_on_the_lo_stage": d, "pick_us_p50": statistics.median(alt_pick)}
        print(f"G1330 W4 {lo}<->{hi} chunk_delta_ms={d:+.3f} (the lo-stage chunk after a switch vs the same stage in a row) "
              f"pick_us_p50={statistics.median(alt_pick):.1f}")
    out["w4"] = sw

    # ---- W5: a D stand-in's round per P stage ----------------------------------------------------------------------
    dd = torch.randn(64, 5120, dtype=torch.bfloat16, device=dev)
    dw = torch.randn(5120, 5120, dtype=torch.bfloat16, device=dev)
    dbuf = torch.empty(64, 5120, dtype=torch.bfloat16, device=dev)
    ds = torch.cuda.Stream()
    with torch.cuda.stream(ds):
        for _ in range(3):
            torch.matmul(dd, dw, out=dbuf)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=ds):
        x = dd
        for _ in range(128):
            x = torch.matmul(x, dw)
    torch.cuda.synchronize()

    def d_solo():
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(ds):
            e0.record()
            g.replay()
            e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1)

    solo = statistics.median([d_solo() for _ in range(20)])
    w5 = {"d_solo_ms": solo}
    print(f"G1330 W5 d_standin_solo_ms={solo:.2f}")
    for i in served:
        rounds = []
        for _ in range(12):
            seq += 1
            act.apply(G.Weg2DualGreenRung(seq, i, int(round(fractions[i] * 1e6))))
            ctx, gc = act.pick(sched)
            with ctx:
                sched.forward_stream.wait_stream(sched.schedule_stream)
                if gc is not None:
                    gc.wait_stream(sched.forward_stream)
                chunk()
                chunk()
                if gc is not None:
                    sched.forward_stream.wait_stream(gc)
            rounds.append(d_solo())            # D's round issued while the P chunks run
            torch.cuda.synchronize()
        w5[i] = {"f": fractions[i], "d_round_ms_p50": statistics.median(rounds), "x_solo": statistics.median(rounds) / solo}
        print(f"G1330 W5 stage={i} f={fractions[i]:.2f} d_round_ms_p50={w5[i]['d_round_ms_p50']:.2f} x_solo={w5[i]['x_solo']:.2f}")
    out["w5"] = w5
    free2 = torch.cuda.mem_get_info()[0]
    out["w6_after_walk_mib"] = (free0 - free2) / 2**20
    print(f"G1330 W6 after the walk {(free0 - free2) / 2**20:.1f} MiB below the start")
    print("G1330 STATUS", act.status_line())
    _dump(args.out, out)
    return 0


def _dump(path: str, out: dict) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print("G1330 wrote", path)


if __name__ == "__main__":
    sys.exit(main())
