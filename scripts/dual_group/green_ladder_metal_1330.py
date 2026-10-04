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
  W0  (S1) MPS state before anything: ``is_mps_client`` (get_server_list), the env; after a hang the server list again.
  W2c (S2) CORRECTNESS per stage: the same seeded inputs through a GEMM chain, a Triton add+norm kernel, sglang's
      Triton l2norm, FlashInfer single prefill and the fla gated-delta-rule chunk kernel (each only if importable,
      a missing one is a printed SKIP) on every green stream, compared with stage 100 % (bitwise flag + max abs diff
      against a bf16 tolerance); a kernel that hangs ends the process with exit 3 under the same watchdog.
  W7  (S3) ALLOCATOR per stage: activation-like allocations (6 x 64 MiB torch.empty) inside the stage's stream
      context, with a filler that leaves ~0.6 GiB free: memory_reserved, num_alloc_retries, OOM, seconds per stage
      change, per stage.
  W8  (S4) RACE: a producer on the schedule stream (fill pattern), the consumer through the hook pattern on the
      green stream with a checksum, the tensor freed / re-allocated / overwritten at once, 1000 x; plus a consumer
      on forward_stream after forward.wait(gc); plus the same on rung 0 (the hazard exists without green contexts:
      the caching allocator does not know a second stream read the block) and a record_stream control.
  W1b (S6) a second probe shape next to the boot probe's: the old 2048x4096x4096 (4 waves on 128 AND 170 SM, the
      review's point) beside the metal probe's 4096x5120x8192 (~12 waves).

Time budget per card: --budget-s (default 840): a section that would start past it prints SKIP-BUDGET.

NOT covered here (needs the real boot, see the report): the prefill graph going eager per stage, the P
allocator pools per green stream with real activations, PP wire uniformity across the three stages, the
Barlink collectives next to a narrow stream, D rounds with the real D.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def mps_state() -> dict:
    """S1: is this process an MPS client? (the probe's ``mps_evidence``: get_server_list rc/servers)"""
    out = {"pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "env_active_thread_percentage":
           os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), "env_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY")}
    for cmd in ("get_server_list", "get_client_list"):
        try:
            r = subprocess.run(["nvidia-cuda-mps-control"], input=cmd + "\n", capture_output=True, text=True, timeout=5,
                               env=dict(os.environ))
            out[cmd] = {"rc": r.returncode, "out": r.stdout.strip()[:400], "err": r.stderr.strip()[:200]}
        except Exception as e:  # noqa: BLE001 - evidence only
            out[cmd] = {"rc": None, "out": "", "err": f"{type(e).__name__}: {e}"}
    srv = out["get_server_list"]
    out["is_mps_client"] = bool(srv["rc"] == 0 and any(t.isdigit() for t in srv["out"].split()))
    return out


try:  # S2: one Triton kernel of our own (add + RMS-norm), defined at module level so triton can read its source
    import triton
    import triton.language as tl

    @triton.jit
    def _add_norm_kernel(x_ptr, y_ptr, o_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        z = x + y
        var = tl.sum(z * z, axis=0) / n_cols
        tl.store(o_ptr + row * n_cols + cols, (z * tl.rsqrt(var + eps)).to(tl.bfloat16), mask=mask)
except Exception:  # noqa: BLE001 - no triton in this venv: W2c prints a SKIP
    _add_norm_kernel = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--chunks", type=int, default=200, help="chunks per stage for the needle test")
    ap.add_argument("--budget-s", type=float, default=840.0, help="time budget of the whole run (15 min card window)")
    ap.add_argument("--race-iters", type=int, default=1000)
    ap.add_argument("--filler-free-mib", type=int, default=600, help="W7: VRAM left free by the filler")
    ap.add_argument("--hang-s", type=float, default=20.0, help="per-chunk watchdog")
    ap.add_argument("--m", type=int, default=1024, help="P-like chunk tokens")
    ap.add_argument("--n-gemm", type=int, default=64, help="GEMM launches per P chunk")
    ap.add_argument("--w2c-only", action="store_true",
                    help="1330/W2c fast probe: after W1/W2/W2c write the JSON and exit (rc 1 when W2c is bad), skip W3-W8 (~5 min)")
    ap.add_argument("--require-kernels", default="gemm_chain,flashinfer_prefill,fla_gated_delta_rule",
                    help="W2c kernels that must RUN on every served stage: a SKIP of one of them counts as bad (a SKIP is not a pass)")
    args = ap.parse_args()
    if os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
        print("G1330 REFUSED CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set: the ladder cannot lift a static percentage")
        return 2

    T0 = time.perf_counter()

    def left() -> float:
        return args.budget_s - (time.perf_counter() - T0)

    mps0 = mps_state()
    print(f"G1330 W0 is_mps_client={mps0['is_mps_client']} pipe={mps0['pipe']} env_pct={mps0['env_active_thread_percentage']} "
          f"server_list_rc={mps0['get_server_list']['rc']} out={mps0['get_server_list']['out']!r} err={mps0['get_server_list']['err']!r}")
    if not mps0["is_mps_client"]:
        print("G1330 W0 WARN not an MPS client (no server in the pipe dir): the probe numbers are NOT the dual-boot's conditions")

    import torch

    from sglang.srt.weg2 import dual_green as G
    from sglang.srt.weg2 import dual_share as S

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    out: dict = {"card": torch.cuda.get_device_name(0), "mps_pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
                 "w0_mps": mps0}
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
    # W1b (S6): the boot probe's shape (now the metal probe's) beside the OLD one that timed 128 and 170 SM identically
    shapes = {"boot_probe_shape": G.PROBE_SHAPE_DEFAULT, "old_2048x4096x4096": (2048, 4096, 4096),
              "metal_probe_4096x5120x8192": (4096, 5120, 8192)}
    w1b = {}
    info = ladder.info
    for label, shp in shapes.items():
        try:
            t_full = be.time_stream(None, shape=shp)
        except Exception as e:  # noqa: BLE001
            print(f"G1330 W1b {label} baseline failed: {e}")
            continue
        tiles = (shp[0] // 128) * (shp[2] // 128)
        w1b[label] = {"shape": shp, "ctas_128x128": tiles, "t_full_ms": t_full, "rungs": {}}
        for i, r in sorted(ladder.rungs.items()):
            t = be.time_stream(r.stream, shape=shp)
            ratio = t / t_full
            sm_ratio = info.sm_total / r.sm
            w1b[label]["rungs"][i] = {"sm": r.sm, "time_ratio": ratio, "sm_ratio": sm_ratio, "effect": ratio / sm_ratio,
                                      "waves_full": -(-tiles // info.sm_total), "waves_masked": -(-tiles // r.sm)}
            print(f"G1330 W1b {label} stage={i} sm={r.sm} time_ratio={ratio:.3f} sm_ratio={sm_ratio:.3f} effect={ratio / sm_ratio:.3f} "
                  f"waves {-(-tiles // info.sm_total)} -> {-(-tiles // r.sm)}")
    out["w1b"] = w1b
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
                print("G1330 HANG MPS after the hang:", json.dumps(mps_state()))
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

    # ---- W2c (S2): correctness per stage with real kernels ----------------------------------------------------------
    def next_seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    def wait_done(done, rung, what):
        t_start = time.perf_counter()
        while not done.query():
            if time.perf_counter() - t_start > args.hang_s:
                print(f"G1330 HANG {what} stage={rung} f={fractions[rung]:.2f} sm={act.active_sm} after {args.hang_s}s")
                print("G1330 HANG MPS after the hang:", json.dumps(mps_state()))
                sys.stdout.flush()
                os._exit(3)
            time.sleep(0.0005)
        torch.cuda.synchronize()

    def on_stage(rung, fn, what="kernel"):
        """The hook of _pp_launch_batch around ``fn`` (inputs made beforehand on the default stream)."""
        act.apply(G.Weg2DualGreenRung(next_seq(), rung, int(round(fractions[rung] * 1e6))))
        ctx, gc = act.pick(sched)
        with ctx:
            sched.forward_stream.wait_stream(sched.schedule_stream)
            if gc is not None:
                gc.wait_stream(sched.forward_stream)
            res = fn()
            done = torch.cuda.Event()
            done.record()
            if gc is not None:
                sched.forward_stream.wait_stream(gc)
        wait_done(done, rung, what)
        return res

    gen = torch.Generator(device=dev)
    gen.manual_seed(1330)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=dev, generator=gen)

    w2c: dict = {}
    cases = {}
    x1, w1, w2m = rnd(args.m, 5120), rnd(5120, 5120) * 0.02, rnd(5120, 5120) * 0.02
    cases["gemm_chain"] = lambda: ((x1 @ w1) @ w2m).clone()
    if _add_norm_kernel is not None:
        xa, ya = rnd(2048, 5120), rnd(2048, 5120)

        def add_norm():
            o = torch.empty_like(xa)
            _add_norm_kernel[(xa.shape[0],)](xa, ya, o, xa.shape[1], 1e-6, BLOCK=8192)
            return o

        cases["triton_add_norm"] = add_norm
    else:
        print("G1330 W2c SKIP triton_add_norm: triton not importable")
    try:
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        xl = rnd(2048, 128)
        cases["sglang_triton_l2norm"] = lambda: l2norm_fwd(xl).clone()
    except Exception as e:  # noqa: BLE001
        print(f"G1330 W2c SKIP sglang_triton_l2norm: {type(e).__name__}: {e}")
    try:
        import flashinfer

        fq, fk, fv = rnd(1024, 32, 128), rnd(1024, 8, 128), rnd(1024, 8, 128)
        cases["flashinfer_prefill"] = lambda: flashinfer.single_prefill_with_kv_cache(fq, fk, fv, causal=True).clone()
    except Exception as e:  # noqa: BLE001
        print(f"G1330 W2c SKIP flashinfer_prefill: {type(e).__name__}: {e}")
    try:
        import torch.nn.functional as F

        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

        T_g, Hg_g, H_g, K_g = 1024, 4, 8, 128      # q/k heads Hg, v heads H = 2 Hg (the GQA branch i_h // (H // Hg))
        gq = rnd(1, T_g, Hg_g, K_g)
        gk = F.normalize(rnd(1, T_g, Hg_g, K_g), p=2, dim=-1)
        gv = rnd(1, T_g, H_g, K_g)
        gb = rnd(1, T_g, H_g).sigmoid()
        gg = F.logsigmoid(rnd(1, T_g, H_g))
        # the contract of gdn_triton.TritonGDNKernel.extend: recurrent state POOL [slots,H,V,K], int32 slot index per
        # sequence, int32 cu_seqlens; the kernel loads ``initial_state_indices`` unconditionally (None was the 04.10.
        # 12:28Z CompilationError that turned W2c into a SKIP). The kernel updates the pool in place -> fresh pool per call.
        gpool0 = torch.zeros(2, H_g, K_g, K_g, dtype=torch.float32, device=dev)
        gidx = torch.zeros(1, dtype=torch.int32, device=dev)
        gcu = torch.tensor([0, T_g], dtype=torch.int32, device=dev)

        def gdn():
            r = chunk_gated_delta_rule(gq, gk, gv, gg, gb, initial_state=gpool0.clone(), initial_state_indices=gidx,
                                       cu_seqlens=gcu, head_first=False, use_qk_l2norm_in_kernel=True)
            return (r[0] if isinstance(r, (tuple, list)) else r).clone()

        cases["fla_gated_delta_rule"] = gdn
    except Exception as e:  # noqa: BLE001
        print(f"G1330 W2c SKIP fla_gated_delta_rule: {type(e).__name__}: {e}")
    torch.cuda.synchronize()
    w2c_bad = 0
    for name, fn in cases.items():
        try:
            ref = on_stage(0, fn, name)                         # first call on the primary stream also compiles
        except Exception as e:  # noqa: BLE001
            print(f"G1330 W2c SKIP {name}: reference failed {type(e).__name__}: {e}")
            continue
        for i in served[1:]:
            try:
                got = on_stage(i, fn, name)
            except Exception as e:  # noqa: BLE001
                print(f"G1330 W2c FAIL {name} stage={i}: {type(e).__name__}: {e}")
                w2c_bad += 1
                continue
            rf, gf = ref.float(), got.float()
            md = float((rf - gf).abs().max())
            bw = bool(torch.equal(ref, got))
            ok = bool(torch.allclose(rf, gf, rtol=2e-2, atol=2e-2)) and bool(torch.isfinite(gf).all())
            w2c_bad += 0 if ok else 1
            w2c.setdefault(name, {})[i] = {"bitwise": bw, "max_abs_diff": md, "ok": ok}
            print(f"G1330 W2c {name} stage={i} sm={ladder.rungs[i].sm} bitwise={bw} max_abs_diff={md:.3e} within_tol={ok}")
    for need in [k for k in args.require_kernels.split(",") if k]:
        got = w2c.get(need, {})
        if any(i not in got for i in served[1:]):
            print(f"G1330 W2c FAIL-NOT-RUN {need}: ran on stages {sorted(got)} of {served[1:]} (a SKIP is not a pass)")
            w2c_bad += 1
    print(f"G1330 W2c bad={w2c_bad} (must be 0) kernels={list(cases)}")
    out["w2c"] = w2c
    out["w2c_bad"] = w2c_bad
    if args.w2c_only:
        print(f"G1330 W2c-ONLY {'FAIL' if w2c_bad else 'PASS'} bad={w2c_bad} kernels={list(cases)} stages={served}")
        _dump(args.out, out)
        return 1 if w2c_bad else 0

    # ---- W3: needle (hang) test per stage ---------------------------------------------------------------------
    for i in served:
        if left() < 300:
            print(f"G1330 W3 SKIP-BUDGET stage={i} ({left():.0f}s left)")
            continue
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

    # ---- W7 (S3): allocator per stage with a filler that leaves ~0.6 GiB free ----------------------------------------
    if left() > 90:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_b, _tot = torch.cuda.mem_get_info()
        fill = max(0, free_b - args.filler_free_mib * 2**20)
        filler = torch.empty(fill, dtype=torch.uint8, device=dev) if fill else None
        print(f"G1330 W7 filler {fill / 2**20:.0f} MiB placed, {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB free")
        w7 = {}
        for i in served + served[-2::-1]:
            st0 = torch.cuda.memory_stats()
            r0 = int(st0.get("num_alloc_retries", 0))
            t0 = time.perf_counter()
            act.apply(G.Weg2DualGreenRung(next_seq(), i, int(round(fractions[i] * 1e6))))
            ctx, gc = act.pick(sched)
            t_switch_us = (time.perf_counter() - t0) * 1e6
            oom = None
            t1 = time.perf_counter()
            try:
                with ctx:
                    sched.forward_stream.wait_stream(sched.schedule_stream)
                    if gc is not None:
                        gc.wait_stream(sched.forward_stream)
                    for it in range(30):
                        ts = [torch.empty(64 * 2**20, dtype=torch.uint8, device=dev) for _ in range(6)]
                        for t in ts:
                            t.fill_(it & 0xFF)
                        del ts
                    done = torch.cuda.Event()
                    done.record()
                    if gc is not None:
                        sched.forward_stream.wait_stream(gc)
                wait_done(done, i, "allocator")
            except torch.cuda.OutOfMemoryError as e:
                oom = str(e)[:160]
            ms = (time.perf_counter() - t1) * 1000.0
            st1 = torch.cuda.memory_stats()
            row = {"stage": i, "reserved_mib": torch.cuda.memory_reserved() >> 20, "retries_delta": int(st1.get("num_alloc_retries", 0)) - r0,
                   "switch_us": t_switch_us, "alloc_ms_30_iters": ms, "oom": oom}
            w7.setdefault(i, []).append(row)
            print(f"G1330 W7 stage={i} reserved_mib={row['reserved_mib']} num_alloc_retries_delta={row['retries_delta']} "
                  f"switch_us={t_switch_us:.1f} alloc_ms_30x6x64MiB={ms:.1f} oom={oom}")
        out["w7"] = w7
        del filler
        torch.cuda.empty_cache()
    else:
        print(f"G1330 W7 SKIP-BUDGET ({left():.0f}s left)")

    # ---- W8 (S4): the cross-stream lifetime race ---------------------------------------------------------------------------
    if left() > 60 and len(served) > 1:
        mid = served[1]
        n_el = 8 * 2**20
        big_a, big_w = rnd(args.m, 5120), rnd(5120, 5120)

        def race(variant: str, iters: int) -> int:
            res = torch.zeros(iters, dtype=torch.int64, device=dev)
            rung = 0 if variant == "rung0_forward_stream" else mid
            for i in range(iters):
                with torch.cuda.stream(sch):                   # the producer: the schedule stream's allocation + fill
                    t = torch.empty(n_el, dtype=torch.int32, device=dev)
                    t.fill_(i + 1)
                act.apply(G.Weg2DualGreenRung(next_seq(), rung, int(round(fractions[rung] * 1e6))))
                ctx, gc = act.pick(sched)
                with ctx:
                    sched.forward_stream.wait_stream(sched.schedule_stream)
                    if gc is not None:
                        gc.wait_stream(sched.forward_stream)
                    if variant == "fwd_after_gc":
                        torch.matmul(big_a, big_w)             # work on the green stream first ...
                    else:
                        for _ in range(3):
                            torch.matmul(big_a, big_w)         # a slow consumer: the host runs ahead of it
                        res[i] = t.sum(dtype=torch.int64)
                    if variant == "gc_hook_record_stream" and gc is not None:
                        t.record_stream(gc)
                    if gc is not None:
                        sched.forward_stream.wait_stream(gc)
                if variant == "fwd_after_gc":                  # ... then the consumer on forward_stream after forward.wait(gc)
                    with sched.forward_stream_ctx:
                        for _ in range(3):
                            torch.matmul(big_a, big_w)
                        res[i] = t.sum(dtype=torch.int64)
                del t                                           # freed at once: the next iteration's producer re-allocates it
            torch.cuda.synchronize()
            want = (torch.arange(iters, device=dev, dtype=torch.int64) + 1) * n_el
            return int((res != want).sum())

        w8 = {}
        for variant in ("rung0_forward_stream", "gc_hook", "fwd_after_gc", "gc_hook_record_stream"):
            if left() < 30:
                print(f"G1330 W8 SKIP-BUDGET {variant}")
                continue
            bad = race(variant, args.race_iters)
            w8[variant] = bad
            print(f"G1330 W8 {variant} iters={args.race_iters} mismatches={bad} "
                  f"({'RACE' if bad else 'clean'}; rung0_forward_stream shows whether the hazard exists WITHOUT green contexts, "
                  "gc_hook_record_stream is the control with the fix)")
        out["w8"] = w8
    else:
        print(f"G1330 W8 SKIP-BUDGET ({left():.0f}s left)")
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
