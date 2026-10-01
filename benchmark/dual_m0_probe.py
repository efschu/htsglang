#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dual-model M0 probe: four processes per card, no model.

    python benchmark/dual_m0_probe.py --out /out/m0.json [--cards 0,1,2] [--cycles 3]

What it answers (concept DUAL-MODEL-FLIP-KONZEPT-1001.md, section 8):

1. **Idle CUDA contexts.** Twelve worker processes, four per card (one per
   group slot: 27B-P, 27B-D, NF-P, NF-D), each creating a context with
   ``torch.zeros(1)``. NVML used MiB per card is sampled after every slot,
   so the per-context cost is the per-slot delta.
2. **BAR1 with four barlink groups.** Each slot is a gloo world of three ranks
   (one per card) and builds ``BarlinkBar1Transport`` objects with today's
   window sizes (P: world 24 + pp 96 MiB, D: world 16 + tp 32 + dcp 40 MiB,
   boot logs 10010415 / 10010340). NVML BAR1 Used is sampled after every
   group. On a 3080 (256 MiB, no ReBAR) the third group is expected to be
   refused; the refusal text is the data point.
3. **Model-flip BAR1 swap, upper bound.** ``--cycles`` times: close both
   groups of one model, build both groups of the other, and back. Close and
   build wall times per group are the upper bound of a re-attach (a full
   rebuild incl. fd exchange; the VA-stable detach of section 8.3 is cheaper
   and is proven by a later probe).
4. **Deep-sleep floor.** After the last close every worker empties the torch
   cache; NVML used per card / 4 is the floor a model in deep sleep can reach
   per process (context + gloo + loaded extension, no windows).

Nothing here loads a model or touches a server. Every collective runs with
a 60 s gloo timeout, so a refused group cannot hang the probe. Output: one
``DUALM0 ...`` line per measurement on stdout, plus the full JSON at --out.
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

MIB = 1 << 20

#: slot -> (name, [(group name, window MiB)]); sizes from the boot logs.
SLOTS = [
    ("27B-P", [("world", 24), ("pp", 96)]),
    ("27B-D", [("world", 16), ("tp", 32), ("dcp", 40)]),
    ("NF-P", [("world", 24), ("pp", 96)]),
    ("NF-D", [("world", 16), ("tp", 32), ("dcp", 40)]),
]
GLOO_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def worker(card: int, rank: int, slot: int, conn) -> None:
    """One process: rank ``rank`` of slot ``slot``'s world, on ``card``."""
    import torch

    state = {"transports": [], "port": None}

    def reply(**kw):
        conn.send(kw)

    def init(port: int) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
            world_size=3, timeout=datetime.timedelta(seconds=GLOO_TIMEOUT_S))
        state["port"] = port

    def build(windows) -> List[dict]:
        import torch.distributed as dist
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            BarlinkBar1Transport,
        )

        out = []
        dev = torch.device("cuda", card)
        for name, mib in windows:
            t0 = time.time()
            rec = {"group": name, "window_mib": mib, "ok": False}
            try:
                pg = dist.new_group(
                    ranks=[0, 1, 2], backend="gloo",
                    timeout=datetime.timedelta(seconds=GLOO_TIMEOUT_S))
                t = BarlinkBar1Transport(pg, dev, mib * MIB, enabled=True, group=name)
                proofs = t.byte_proof_all()
                rec["proof"] = {str(k): bool(v) for k, v in proofs.items()}
                rec["ok"] = all(proofs.values())
                state["transports"].append(t)
            except BaseException as e:  # the refusal IS the data point
                rec["error"] = f"{type(e).__name__}: {e}"[:600]
            rec["ms"] = round((time.time() - t0) * 1000, 1)
            out.append(rec)
            if not rec["ok"]:
                break  # a broken group poisons the next collective
        torch.cuda.synchronize()
        return out

    def close() -> dict:
        t0 = time.time()
        errs = []
        for t in state["transports"]:
            try:
                t.close()
            except BaseException as e:
                errs.append(f"{type(e).__name__}: {e}"[:300])
        state["transports"].clear()
        torch.cuda.synchronize()
        return {"ms": round((time.time() - t0) * 1000, 1), "errors": errs}

    try:
        while True:
            cmd, arg = conn.recv()
            try:
                if cmd == "ctx":
                    torch.cuda.set_device(card)
                    torch.zeros(1, device=torch.device("cuda", card))
                    torch.cuda.synchronize()
                    reply(ok=True)
                elif cmd == "init":
                    init(arg)
                    reply(ok=True)
                elif cmd == "build":
                    res = build(arg)
                    reply(ok=all(r["ok"] for r in res) and len(res) == len(arg), res=res)
                elif cmd == "close":
                    reply(ok=True, res=close())
                elif cmd == "empty_cache":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    reply(ok=True, reserved_mib=torch.cuda.memory_reserved(card) / MIB)
                elif cmd == "exit":
                    close()
                    reply(ok=True)
                    return
                else:
                    reply(ok=False, error=f"unknown command {cmd!r}")
            except BaseException as e:
                reply(ok=False, error=f"{type(e).__name__}: {e}"[:600],
                      tb=traceback.format_exc()[-1500:])
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


class Nvml:
    def __init__(self, cards: List[int]):
        import pynvml

        self.n = pynvml
        pynvml.nvmlInit()
        self.cards = cards
        self.h = {c: pynvml.nvmlDeviceGetHandleByIndex(c) for c in cards}
        self.names = {c: _s(pynvml.nvmlDeviceGetName(self.h[c])) for c in cards}

    def sample(self) -> Dict[str, dict]:
        out = {}
        for c in self.cards:
            m = self.n.nvmlDeviceGetMemoryInfo(self.h[c])
            b = self.n.nvmlDeviceGetBAR1MemoryInfo(self.h[c])
            try:
                procs = self.n.nvmlDeviceGetComputeRunningProcesses(self.h[c])
                per = sorted(int((p.usedGpuMemory or 0) / MIB) for p in procs)
            except Exception:
                per = []
            out[str(c)] = {
                "name": self.names[c], "used_mib": round(m.used / MIB),
                "total_mib": round(m.total / MIB),
                "bar1_used_mib": round(b.bar1Used / MIB),
                "bar1_total_mib": round(b.bar1Total / MIB),
                "procs": len(per), "proc_used_mib": per,
            }
        return out


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


class Probe:
    def __init__(self, cards: List[int], port_base: int):
        self.cards = cards
        self.port_base = port_base
        self.port_seq = 0
        self.nv = Nvml(cards)
        self.rec: dict = {"cards": {str(c): self.nv.names[c] for c in cards},
                          "slots": [s[0] for s in SLOTS], "steps": []}
        self.w: Dict[tuple, tuple] = {}  # (slot, rank) -> (proc, conn)

    # -- plumbing ---------------------------------------------------------
    def spawn_slot(self, slot: int) -> None:
        ctx = mp.get_context("spawn")
        for rank, card in enumerate(self.cards):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(card, rank, slot, b), daemon=True)
            p.start()
            self.w[(slot, rank)] = (p, a)

    def call(self, slot: int, cmd: str, arg=None, timeout: float = 120.0) -> List[dict]:
        for rank in range(len(self.cards)):
            self.w[(slot, rank)][1].send((cmd, arg))
        out = []
        deadline = time.time() + timeout
        for rank in range(len(self.cards)):
            conn = self.w[(slot, rank)][1]
            left = max(0.1, deadline - time.time())
            if conn.poll(left):
                out.append(conn.recv())
            else:
                out.append({"ok": False, "error": f"no reply in {timeout:.0f} s"})
        return out

    def step(self, name: str, **extra) -> dict:
        s = {"step": name, "t": round(time.time(), 3), "nvml": self.nv.sample(), **extra}
        self.rec["steps"].append(s)
        cards = " ".join(
            f"c{c}[{v['name'].replace('NVIDIA GeForce ', '')}] used={v['used_mib']} "
            f"bar1={v['bar1_used_mib']}/{v['bar1_total_mib']} procs={v['procs']}"
            for c, v in s["nvml"].items())
        print(f"DUALM0 {name}: {cards}", flush=True)
        for k, v in extra.items():
            print(f"DUALM0   {k}: {json.dumps(v)[:900]}", flush=True)
        return s

    def init_slot(self, slot: int) -> bool:
        self.port_seq += 1
        port = self.port_base + 10 * slot + self.port_seq
        res = self.call(slot, "init", port, timeout=GLOO_TIMEOUT_S + 30)
        return all(r.get("ok") for r in res)

    def build_slot(self, slot: int, timeout: float) -> dict:
        res = self.call(slot, "build", SLOTS[slot][1], timeout=timeout)
        ok = all(r.get("ok") for r in res)
        if not ok:  # a failed collective leaves gloo unusable: fresh world
            self.call(slot, "close")
            self.init_slot(slot)
        return {"ok": ok, "ranks": res}

    def close_slot(self, slot: int) -> dict:
        return {"ranks": self.call(slot, "close")}

    # -- the run ----------------------------------------------------------
    def run(self, cycles: int, first_build_timeout: float) -> dict:
        base = self.step("baseline")
        busy = {c: v["procs"] for c, v in base["nvml"].items() if v["procs"]}
        if busy and not os.environ.get("DUALM0_ALLOW_BUSY"):
            self.rec["refused"] = f"cards not idle: {busy} (set DUALM0_ALLOW_BUSY=1 to measure anyway)"
            print(f"DUALM0 REFUSED {self.rec['refused']}", flush=True)
            return self.rec

        # 1. contexts, slot by slot
        for slot in range(len(SLOTS)):
            self.spawn_slot(slot)
            res = self.call(slot, "ctx", timeout=120)
            self.step(f"ctx+{SLOTS[slot][0]}", ok=[r.get("ok") for r in res])
        for slot in range(len(SLOTS)):
            self.init_slot(slot)
        self.step("ctx+gloo(all 4 slots)")

        # 2. four groups on top of each other
        built = []
        for slot in range(len(SLOTS)):
            t0 = time.time()
            r = self.build_slot(slot, first_build_timeout if slot == 0 else 180)
            self.step(f"build {SLOTS[slot][0]}", result=_brief(r),
                      wall_ms=round((time.time() - t0) * 1000))
            if r["ok"]:
                built.append(slot)

        # 3. model-flip BAR1 swap: 27B (0,1) <-> NF (2,3)
        for slot in list(built):
            self.close_slot(slot)
        built = []
        self.step("all closed")
        flips = []
        for cyc in range(cycles):
            for model, slots in (("27B", (0, 1)), ("NF", (2, 3))):
                rec = {"cycle": cyc, "model": model, "build": {}, "close": {}}
                for slot in slots:
                    t0 = time.time()
                    r = self.build_slot(slot, 180)
                    rec["build"][SLOTS[slot][0]] = {
                        "ok": r["ok"], "wall_ms": round((time.time() - t0) * 1000),
                        "rank_ms": [sum(x.get("ms", 0) for x in rr.get("res", [])) for rr in r["ranks"]],
                        "errors": [e for rr in r["ranks"] for e in [rr.get("error")] if e][:3]
                                  + [x.get("error") for rr in r["ranks"] for x in rr.get("res", []) if x.get("error")][:3],
                    }
                s_up = self.step(f"cycle{cyc} {model} up", build=rec["build"])
                rec["bar1_up"] = {c: v["bar1_used_mib"] for c, v in s_up["nvml"].items()}
                for slot in slots:
                    t0 = time.time()
                    r = self.close_slot(slot)
                    rec["close"][SLOTS[slot][0]] = {
                        "wall_ms": round((time.time() - t0) * 1000),
                        "rank_ms": [rr.get("res", {}).get("ms") for rr in r["ranks"]]}
                s_dn = self.step(f"cycle{cyc} {model} down", close=rec["close"])
                rec["bar1_down"] = {c: v["bar1_used_mib"] for c, v in s_dn["nvml"].items()}
                flips.append(rec)
        self.rec["flips"] = flips

        # 4. deep-sleep floor
        for slot in range(len(SLOTS)):
            self.call(slot, "empty_cache")
        self.step("deep-sleep floor (12 procs, no windows, cache emptied)")

        for slot in range(len(SLOTS)):
            self.call(slot, "exit", timeout=60)
        time.sleep(2)
        self.step("after exit")
        self.rec["summary"] = summarize(self.rec)
        print("DUALM0 SUMMARY " + json.dumps(self.rec["summary"]), flush=True)
        return self.rec


def _brief(r: dict) -> dict:
    return {"ok": r["ok"], "ranks": [
        {"ok": rr.get("ok"), "error": rr.get("error"),
         "res": [{k: x.get(k) for k in ("group", "window_mib", "ok", "ms", "error")}
                 for x in rr.get("res", [])]} for rr in r["ranks"]]}


def summarize(rec: dict) -> dict:
    """Per-card deltas between named steps; pure (unit-testable)."""
    steps = {s["step"]: s["nvml"] for s in rec.get("steps", [])}
    out: dict = {}
    base = steps.get("baseline")
    if not base:
        return out
    names = [n for n in steps if n.startswith("ctx+") and "gloo" not in n]
    prev = base
    per_ctx: Dict[str, List[int]] = {c: [] for c in base}
    for n in names:
        for c in base:
            per_ctx[c].append(steps[n][c]["used_mib"] - prev[c]["used_mib"])
        prev = steps[n]
    out["ctx_delta_mib_per_slot"] = per_ctx
    out["bar1_baseline_mib"] = {c: base[c]["bar1_used_mib"] for c in base}
    for n, v in steps.items():
        if n.startswith("build "):
            out.setdefault("bar1_after_build_mib", {})[n[6:]] = {c: v[c]["bar1_used_mib"] for c in v}
    floor = steps.get("deep-sleep floor (12 procs, no windows, cache emptied)")
    if floor:
        out["deep_sleep_floor_mib_per_proc"] = {
            c: round((floor[c]["used_mib"] - base[c]["used_mib"]) / 4) for c in base}
    flips = rec.get("flips") or []
    if flips:
        b = [v["wall_ms"] for f in flips for v in f["build"].values() if v["ok"]]
        k = [v["wall_ms"] for f in flips for v in f["close"].values()]
        out["group_build_wall_ms"] = {"n": len(b), "min": min(b) if b else None,
                                      "max": max(b) if b else None}
        out["group_close_wall_ms"] = {"n": len(k), "min": min(k) if k else None,
                                      "max": max(k) if k else None}
        out["flip_builds_failed"] = sum(1 for f in flips for v in f["build"].values() if not v["ok"])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cards", default="0,1,2")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--port-base", type=int, default=29810)
    ap.add_argument("--first-build-timeout", type=float, default=360.0,
                    help="slot 0 may pay the barlink extension JIT build")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cards = [int(x) for x in a.cards.split(",")]
    if len(cards) != 3:
        print("DUALM0 REFUSED: exactly three cards (one rank per card per slot)")
        return 2
    t0 = time.time()
    probe = Probe(cards, a.port_base)
    rec = {}
    try:
        rec = probe.run(a.cycles, a.first_build_timeout)
    finally:
        rec = rec or probe.rec
        rec["wall_s"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"DUALM0 wrote {a.out} after {rec['wall_s']} s", flush=True)
        for (p, _c) in probe.w.values():
            if p.is_alive():
                p.terminate()
    return 0 if "refused" not in rec else 3


if __name__ == "__main__":
    sys.exit(main())
