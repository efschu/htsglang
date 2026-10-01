#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Metal proof for the VA-stable BAR1 detach / re-attach (dual-model M0b).

    python benchmark/bar1_reattach_probe.py --out /out [--cards 0,1,2]

One barlink BAR1 group of three ranks (one per card), no model, no server.
A 64 KiB all_reduce is captured into a CUDA graph and replayed with a byte
check BEFORE anything is detached. Then three variants, each followed by
three more replays OF THE SAME GRAPH with fresh inputs, bit-exact:

* ``keep_map``      -- release only the holder attachments, re-hold;
* ``remap``         -- also unregister + PROT_NONE placeholder, then remap at
                       the same host VA and require the same device pointer;
* ``remap_shifted`` -- like ``remap``, but an intruder group (16 MiB) is
                       built while the first one sleeps and stays up during
                       the re-attach, so the BAR1 offsets have to move. This
                       is the dual-model case (the other model's windows sit
                       in the freed range).

Per variant and rank: BAR1 Used on the rank's own card before / detached /
(intruder up) / re-attached, detach and re-attach wall time, how each region
came back (in_place / remap) and the byte verdict of every replay. A refusal
on any rank is agreed collectively and stops the run cleanly (never a kernel
over a mapping that moved).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MIB = 1 << 20
N_ELEMS = (64 << 10) // 4
REPLAYS = 3


def _pattern(n, rank, rnd, dev):
    i = torch.arange(n, dtype=torch.float32, device=dev)
    return (rank + 1) * (rnd + 1) + (i % 97)


def _expected(n, world, rnd, dev):
    i = torch.arange(n, dtype=torch.float32, device=dev)
    return sum((r + 1) * (rnd + 1) for r in range(world)) + world * (i % 97)


def _bar1(card):
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(card)
    return round(pynvml.nvmlDeviceGetBAR1MemoryInfo(h).bar1Used / MIB)


def _agree(ok: bool) -> bool:
    flags = [None] * dist.get_world_size()
    dist.all_gather_object(flags, bool(ok))
    return all(flags)


def worker(rank, cards, port, window_mib, intruder_mib, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(len(cards)),
                      SGLANG_BARLINK_BAR1_GRID_THRESHOLD=str(1 << 40))
    import datetime

    dist.init_process_group("gloo", rank=rank, world_size=len(cards),
                            timeout=datetime.timedelta(seconds=90))
    card = cards[rank]
    torch.cuda.set_device(card)
    dev = torch.device("cuda", card)
    torch.zeros(1, device=dev)
    from sglang.srt.distributed.device_communicators.barlink_bar1 import BarlinkBar1Transport
    from sglang.srt.distributed.device_communicators import barlink_bar1_detach as D

    world = len(cards)
    rec = {"rank": rank, "card": card, "variants": [], "ok": False}
    t = x = None

    def replays(graph, inp, out, base_round, tag):
        verdicts = []
        for k in range(REPLAYS):
            rnd = base_round + k
            inp.copy_(_pattern(N_ELEMS, rank, rnd, dev))
            torch.cuda.synchronize(dev)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(dev)
            bad = int((out != _expected(N_ELEMS, world, rnd, dev)).sum().item())
            verdicts.append(bad)
        rec.setdefault("replays", {})[tag] = verdicts
        return all(v == 0 for v in verdicts)

    try:
        t = BarlinkBar1Transport(dist.group.WORLD, dev, window_mib * MIB, enabled=True, group="world")
        rec["proof_initial"] = all(t.byte_proof_all().values())
        inp = _pattern(N_ELEMS, rank, 0, dev)
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                t.barlink_all_reduce(None, inp)
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = t.barlink_all_reduce(None, inp)
        torch.cuda.synchronize(dev)
        dist.barrier()
        if not _agree(replays(graph, inp, out, 1, "before")):
            raise RuntimeError("graph replay wrong BEFORE any detach -- probe invalid")
        rec["bar1_attached"] = _bar1(card)

        base_round = 10
        for variant in ("keep_map", "remap", "remap_shifted"):
            v = {"variant": variant}
            mode = "keep_map" if variant == "keep_map" else "remap"
            dist.barrier()
            dr = D.detach(t, mode)
            torch.cuda.synchronize(dev)
            dist.barrier()
            v["detach_ms"] = round(dr.ms, 2)
            v["bar1_detached"] = _bar1(card)
            if variant == "remap_shifted":
                g2 = dist.new_group(ranks=list(range(world)), backend="gloo")
                x = BarlinkBar1Transport(g2, dev, intruder_mib * MIB, enabled=True, group="intruder")
                v["intruder_proof"] = all(x.byte_proof_all().values())
                dist.barrier()
                v["bar1_intruder_up"] = _bar1(card)
            dist.barrier()
            t0 = time.time()
            ok = True
            try:
                v["regions"] = D.reattach(t, dr)
            except Exception as e:
                ok = False
                v["refused"] = f"{type(e).__name__}: {e}"[:600]
            v["reattach_ms"] = round((time.time() - t0) * 1000, 2)
            if not _agree(ok):
                v["stopped"] = "a rank refused the re-attach; no kernel runs over it"
                rec["variants"].append(v)
                break
            dist.barrier()
            v["proof_after"] = all(t.byte_proof_all().values())
            v["replays_ok"] = _agree(replays(graph, inp, out, base_round, variant))
            base_round += 10
            v["bar1_reattached"] = _bar1(card)
            if x is not None:
                dist.barrier()
                x.close()
                x = None
                v["bar1_intruder_closed"] = _bar1(card)
            rec["variants"].append(v)
            if not v["replays_ok"]:
                break
        rec["ok"] = all(v.get("replays_ok") for v in rec["variants"]) and len(rec["variants"]) == 3
    except BaseException as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:800]
        rec["tb"] = traceback.format_exc()[-2000:]
    finally:
        for obj in (x, t):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        with open(os.path.join(out_dir, f"reattach_r{rank}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="0,1,2")
    ap.add_argument("--port", type=int, default=29890)
    ap.add_argument("--window-mib", type=int, default=24)
    ap.add_argument("--intruder-mib", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cards = [int(c) for c in a.cards.split(",")]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    try:
        mp.spawn(worker, args=(cards, a.port, a.window_mib, a.intruder_mib, a.out),
                 nprocs=len(cards), join=True)
    except Exception as e:
        print(f"REATTACH spawn: {type(e).__name__}: {e}", flush=True)
    ok = True
    for r in range(len(cards)):
        p = os.path.join(a.out, f"reattach_r{r}.json")
        if not os.path.exists(p):
            print(f"REATTACH r{r}: no record", flush=True)
            ok = False
            continue
        d = json.load(open(p))
        ok &= bool(d.get("ok"))
        print(f"REATTACH r{r} card{d['card']} ok={d.get('ok')} bar1_attached={d.get('bar1_attached')} "
              f"error={d.get('error')}", flush=True)
        for v in d.get("variants", []):
            hows = sorted({x["how"] for x in v.get("regions", [])})
            print(f"REATTACH   {v['variant']}: detach {v.get('detach_ms')} ms, reattach "
                  f"{v.get('reattach_ms')} ms, bar1 {v.get('bar1_detached')} -> "
                  f"{v.get('bar1_intruder_up', '-')} -> {v.get('bar1_reattached')}, how={hows}, "
                  f"proof={v.get('proof_after')} replays_ok={v.get('replays_ok')} "
                  f"refused={v.get('refused')}", flush=True)
    print(f"REATTACH VERDICT {'PASS' if ok else 'FAIL'} after {time.time() - t0:.1f} s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
