#!/usr/bin/env python3
"""QSA-Attention-Mikrobench fuer ein GPU-Fenster (fnFL2 H65).

WAS. Misst die beiden QSA-Prefill-Kernel so, wie der P-Rang sie startet
(dieselben Launcher ``sparse_attn_rows_triton`` / ``sparse_gqa_fwd_interface_
triton``, dieselben Schalter), mit NF-Geometrie (24 q-Koepfe, 2 kv-Koepfe,
head_dim 256, fp8-KV-Pool, Top-k-Breite 2051 = 512 Gruppen x 4 + 3 Tail) und
realistischem Praefix:

* ``rows``    -- Folge-Chunk mit Praefix (16k Query-Zeilen, Praefix 32k..80k,
  Zeilen = Pool-Slots hinter einer zufaelligen 64er-Seitenpermutation);
  Varianten: Launch-Konfiguration (SGLANG_FORCE_QSA_ROWS_CONFIG, H58) x
  fp8-Decode (SGLANG_WEG2_QSA_FP8_DECODE, H65: exp2 | bits | ptx);
* ``prefill`` -- praefixfreier erster Chunk (SGLANG_WEG2_QSA_PREFILL_CONFIG).

Je Variante: Median/Minimum ms je Launch (= je Full-Attention-Layer) ueber
``--reps`` Laeufe nach zwei Aufwaermlaeufen (der erste kompiliert), daraus
ms je Chunk fuer die Stufe dieser Karte (5090: PP0 x 7 Layer, 3080: PP1 x 3 /
PP2 x 2). GLEICHHEIT: gleiche Konfiguration, anderer Decode -> Ausgabe und LSE
muessen BITGLEICH zu exp2 sein (``bit_equal=yes``); andere Konfiguration ->
maximale Abweichung zur Basis (Tabellen-Konfiguration + exp2, Summations-
reihenfolge aendert sich) und zu einer fp32-Referenz auf 16 Stichproben-
Zeilen (``sparse_attn_rows_reference``).

DIE GPU WIRD ERST BEIM START ANGEFASST, und nur so: ``--card`` (NVML-Index)
ist Pflicht, ``--booking`` (gpuq-Fenster-Id) ebenso -- das Skript fragt
``GET /api/v1/bookings/<id>`` und bricht ab, wenn das Fenster nicht LAEUFT oder
die Karte nicht enthaelt (``--no-booking-check`` nur, wenn der Starter das
Fenster selbst haelt). Vor dem ersten CUDA-Aufruf prueft NVML den freien
Speicher der Karte (Bedarf wird ausgegeben); erst danach werden
``CUDA_DEVICE_ORDER=PCI_BUS_ID`` und ``CUDA_VISIBLE_DEVICES=<card>`` gesetzt
und torch geladen. Kein Import dieses Moduls beruehrt CUDA.

Aufruf (Beispiel, 3080 = NVML 0, Buchung ~2 GiB, ~10 min):

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<baum>/python \\
      /spinning/htsglang-gpu/.venv/bin/python -m sglang.srt.weg2.tools.qsa_rows_bench \\
      --card 0 --booking <id> --prefix 32768,81920 \\
      --cfgs table,32/8/2,32/4/2,64/8/2 --decodes exp2,bits,ptx \\
      --out /spinning/evidence-665-f1/qsa_rows_bench_<tag>.jsonl

Ausgabe je Variante eine Zeile ``QSA-ROWS-BENCH ...`` plus JSONL (``--out``).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

LAYERS = {"RTX 5090": {"PP0": 7}, "RTX 3080": {"PP1": 3, "PP2": 2}}
HQ, HKV, D, GROUP_TOPK, RATIO, PAGE = 24, 2, 256, 512, 4, 64
TOPK_COLS = GROUP_TOPK * RATIO + RATIO - 1  # 2051


def _booking_ok(booking: str, card: int) -> str:
    url = f"http://127.0.0.1:8770/api/v1/bookings/{booking}"
    with urllib.request.urlopen(url, timeout=5) as r:
        b = json.loads(r.read().decode())
    state = str(b.get("state"))
    cards = [int(c) for c in (b.get("cards") or [])]
    if state != "running":
        raise SystemExit(f"QSA-ROWS-BENCH REFUSED: booking {booking} state={state!r}, not running")
    if card not in cards:
        raise SystemExit(f"QSA-ROWS-BENCH REFUSED: booking {booking} holds cards {cards}, not {card}")
    end = b.get("end") if isinstance(b.get("end"), dict) else {}
    return (f"booking={booking} state=running cards={cards} until={end.get('utc')} "
            f"left_s={b.get('seconds_left')}")


def _nvml_card(card: int):
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(card)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        return name, mem.free // (1 << 20), mem.total // (1 << 20), [p.pid for p in procs]
    finally:
        pynvml.nvmlShutdown()


def _need_mib(chunk: int, prefix_max: int) -> int:
    q = chunk * HQ * D * 2
    out = 4 * q  # base + this config's first decode + the launch's out (+1 in flight)
    rows = chunk * TOPK_COLS * 4 * 3  # rows + generation scratch
    pool = (prefix_max + chunk + 10 * PAGE) * HKV * D * 2  # fp8 K + V
    kv16 = 2 * chunk * HKV * D * 2  # prefill case: bf16 k/v of the chunk
    return int((q + out + rows + pool + kv16) / (1 << 20)) + 512


def _selection(torch, positions, pattern, gen, dev):
    """[n, 2051] logical token positions (valid first, -1 padded) as QSA's
    top-k would hand them over: 4 sink groups + 32 local groups + 476 far
    groups; 'shared': the far groups of 64 consecutive queries come from one
    candidate set of 952 (neighbours share ~half), 'random': independent,
    'same': identical for the 64. Queries that see <= 512 groups take all."""
    n = positions.numel()
    visible = (positions + 1) // RATIO  # complete groups
    far_n = GROUP_TOPK - 4 - 32
    blk = 64
    nb = (n + blk - 1) // blk
    lo = 4
    hi_q = (visible - 32).clamp(min=lo + 1)
    hi_b = hi_q.view(-1)[torch.arange(nb, device=dev) * blk]  # block's first query: smallest hi
    span = (hi_b - lo).clamp(min=1)
    if pattern == "random":
        u = torch.rand(n, far_n, device=dev, generator=gen)
        far = lo + (u * (hi_q - lo).clamp(min=1).unsqueeze(1)).long()
    else:
        cand_n = far_n if pattern == "same" else 2 * far_n
        u = torch.rand(nb, cand_n, device=dev, generator=gen)
        cand = lo + (u * span.unsqueeze(1)).long()  # [nb, cand_n]
        if pattern == "same":
            far = cand.repeat_interleave(blk, dim=0)[:n]
        else:
            pick = torch.rand(n, cand_n, device=dev, generator=gen).argsort(dim=1)[:, :far_n]
            far = cand.repeat_interleave(blk, dim=0)[:n].gather(1, pick)
    sink = torch.arange(4, device=dev).expand(n, 4)
    local = (visible.unsqueeze(1) - 32 + torch.arange(32, device=dev)).clamp(min=0)
    groups = torch.cat([sink, local, far], dim=1)  # [n, 512]
    # short rows: all visible groups, in order
    allg = torch.arange(GROUP_TOPK, device=dev).expand(n, GROUP_TOPK)
    short = visible.unsqueeze(1) <= GROUP_TOPK
    groups = torch.where(short, torch.where(allg < visible.unsqueeze(1), allg, -1), groups)
    tok = groups.unsqueeze(-1) * RATIO + torch.arange(RATIO, device=dev)
    tok = torch.where(groups.unsqueeze(-1) >= 0, tok, -1).reshape(n, GROUP_TOPK * RATIO)
    tail_start = visible * RATIO
    tail = tail_start.unsqueeze(1) + torch.arange(RATIO - 1, device=dev)
    tail = torch.where(tail <= positions.unsqueeze(1), tail, -1)
    sel = torch.cat([tok, tail], dim=1)
    order = torch.where(sel >= 0, 0, 1).argsort(dim=1, stable=True)
    return sel.gather(1, order).to(torch.int32)


def _time(torch, fn, reps):
    fn()
    fn()  # the first call compiles
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), min(ts)


def _cfg_env(cfg: str) -> str:
    return "" if cfg == "table" else f"inf={cfg}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--card", type=int, required=True, help="NVML index (0/2 = 3080, 1 = 5090)")
    ap.add_argument("--booking", default="", help="gpuq booking id (must be running and hold --card)")
    ap.add_argument("--no-booking-check", action="store_true")
    ap.add_argument("--kernels", default="rows,prefill")
    ap.add_argument("--prefix", default="32768,81920", help="prefix lengths for the rows kernel")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--cfgs", default="table,32/8/2,32/4/2,64/8/2")
    ap.add_argument("--decodes", default="exp2,bits,ptx")
    ap.add_argument("--prefill-cfgs", default="table,32/8/2,32/4/2,64/8/2")
    ap.add_argument("--pattern", default="shared", choices=("shared", "random", "same"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=65)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    prefixes = [int(x) for x in a.prefix.split(",") if x]
    if a.no_booking_check:
        btxt = "booking=unchecked(--no-booking-check)"
    elif not a.booking:
        raise SystemExit("QSA-ROWS-BENCH REFUSED: --booking <gpuq id> is required (or --no-booking-check)")
    else:
        btxt = _booking_ok(a.booking, a.card)
    name, free, total, pids = _nvml_card(a.card)
    need = _need_mib(a.chunk, max(prefixes or [0]))
    print(f"QSA-ROWS-BENCH card={a.card} name={name!r} free={free}/{total} MiB need~{need} MiB "
          f"other_pids={pids} {btxt}", flush=True)
    if free < need + 1024:
        raise SystemExit(f"QSA-ROWS-BENCH REFUSED: {free} MiB free < {need} + 1024 (context) MiB")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.card)
    import torch

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.qsa import sparse_attn as sa

    dev = torch.device("cuda", 0)
    major, minor = torch.cuda.get_device_capability(dev)
    arch = major * 10 + minor
    short = "RTX 5090" if "5090" in name else "RTX 3080" if "3080" in name else name
    stages = LAYERS.get(short, {})
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    scale = 1.0 / (D ** 0.5)
    out_f = open(a.out, "a") if a.out else None

    def emit(rec):
        per_chunk = " ".join(f"{s}={rec['ms_med'] * n:.0f}ms" for s, n in stages.items())
        line = (f"QSA-ROWS-BENCH kernel={rec['kernel']} arch=sm{arch} prefix={rec['prefix']} "
                f"chunk={a.chunk} pattern={a.pattern} cfg={rec['cfg']} decode={rec['decode']} "
                f"ms_med={rec['ms_med']:.2f} ms_min={rec['ms_min']:.2f} vs_base={rec['vs_base']:.3f}x "
                f"bit_equal={rec['bit_equal']} max_abs_vs_base={rec['max_abs_vs_base']:.3g} "
                f"lse_max_abs_vs_base={rec['lse_max_abs_vs_base']:.3g} ref_max_abs={rec['ref_max_abs']:.3g} "
                f"per_chunk[{per_chunk}]")
        print(line, flush=True)
        if out_f:
            rec.update(dict(card=a.card, name=name, arch=arch, chunk=a.chunk, pattern=a.pattern,
                            t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    kernels = [k for k in a.kernels.split(",") if k]
    cfgs = [c for c in a.cfgs.split(",") if c]
    decodes = [d for d in a.decodes.split(",") if d]

    if "rows" in kernels:
        for prefix in prefixes:
            total_tok = prefix + a.chunk
            pages = (total_tok + PAGE - 1) // PAGE
            perm = torch.randperm(pages + 8, device=dev, generator=gen)[:pages] + 1  # page 0 = padding
            pool_rows = (pages + 9) * PAGE
            kp = (torch.randn(pool_rows, HKV, D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
            vp = (torch.randn(pool_rows, HKV, D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
            q = torch.randn(a.chunk, HQ, D, device=dev, generator=gen).to(torch.bfloat16)
            pos = prefix + torch.arange(a.chunk, device=dev)
            logical = _selection(torch, pos, a.pattern, gen, dev)
            safe = logical.clamp(min=0).long()
            slots = perm[safe // PAGE] * PAGE + safe % PAGE
            rows = torch.where(logical >= 0, slots, -1).to(torch.int32).contiguous()
            del safe, slots, logical
            sample = torch.randperm(a.chunk, device=dev, generator=gen)[:16]
            ref, ref_lse = sa.sparse_attn_rows_reference(q[sample], kp, vp, rows[sample], scale)
            base = None
            for cfg in cfgs:
                first = None  # this config's first decode: the bit reference of the others
                for decode in decodes:
                    with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(_cfg_env(cfg)), \
                            envs.SGLANG_WEG2_QSA_FP8_DECODE.override(decode):
                        res = {}

                        def run():
                            res["o"] = sa.sparse_attn_rows_triton(q, kp, vp, rows, scale)

                        med, mn = _time(torch, run, a.reps)
                    out, lse = res.pop("o")
                    if base is None:
                        base = (med, out.clone(), lse.clone())
                    if first is None:
                        first = (decode, out.clone(), lse.clone())
                        bit = f"ref({decode})"
                    else:
                        same = torch.equal(out, first[1]) and torch.equal(lse, first[2])
                        bit = ("yes" if same else "NO") + f"_vs_{first[0]}"
                    emit(dict(kernel="rows", prefix=prefix, cfg=cfg, decode=decode, ms_med=med, ms_min=mn,
                              vs_base=med / base[0], bit_equal=bit,
                              max_abs_vs_base=float((out.float() - base[1].float()).abs().max()),
                              lse_max_abs_vs_base=float((lse - base[2]).abs().max()),
                              ref_max_abs=float((out[sample].float() - ref.float()).abs().max())))
                    del out, lse
                del first
                torch.cuda.empty_cache()
            del kp, vp, q, rows, base
            torch.cuda.empty_cache()

    if "prefill" in kernels:
        n = a.chunk
        q = torch.randn(n, HQ, D, device=dev, generator=gen).to(torch.bfloat16)
        k = torch.randn(n, HKV, D, device=dev, generator=gen).to(torch.bfloat16)
        v = torch.randn(n, HKV, D, device=dev, generator=gen).to(torch.bfloat16)
        idx = _selection(torch, torch.arange(n, device=dev), a.pattern, gen, dev)
        cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
        base = None
        for cfg in [c for c in a.prefill_cfgs.split(",") if c]:
            with envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.override(_cfg_env(cfg)):
                res = {}

                def run():
                    res["o"] = sa.sparse_gqa_fwd_interface_triton(q, k, v, n, idx, cu, scale)

                med, mn = _time(torch, run, a.reps)
            out = res.pop("o")
            is_base = base is None
            if is_base:
                base = (med, out.clone())
            emit(dict(kernel="prefill", prefix=0, cfg=cfg, decode="-", ms_med=med, ms_min=mn,
                      vs_base=med / base[0], bit_equal="ref(table)" if is_base else "-",
                      max_abs_vs_base=float((out.float() - base[1].float()).abs().max()),
                      lse_max_abs_vs_base=0.0, ref_max_abs=float("nan")))
            del out
    if out_f:
        out_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
