# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NCCL card-to-card per ORDERED pair: the reference "how it would run without barlink".

Order 1006 (user addition 06.10. ~14:40Z).  The D2D table of the hardware profile carries three ways per ordered pair side by
side: barlink BAR1 direct (``bar1_probe``), host staging through pinned memory (``card_probe``) and **NCCL point-to-point** (this
module).  NCCL is the comparison measure: without peer access NCCL moves the bytes through the host (SHM) or whatever it picks;
this probe does not force a transport and does not switch one off -- the child environment is the caller's, plus ``NCCL_DEBUG=INFO``
**in the child only**, so the transport NCCL really chose ("via P2P/CUMEM", "via SHM/direct/direct", ...) can be read from its
own log and quoted in the pair's note.

Method (one pair-run = two child processes, one per card, found by NVML UUID exactly as in ``bar1_probe``):

* a NCCL group of size 2; ``dist.send`` / ``dist.recv`` of 64 MiB (uint8, CUDA) -- **rate** = bytes the RECEIVER got / its wall time
  of 4 back-to-back transfers, median of 5 repeats, both directions (A->B then B->A) in the same pair-run;
* **latency** = 4 kB ping-pong started by the SENDER of the direction, round trip / 2, median of 200 (one-way delivery time
  including the receiver's turn-around: a different quantity from a posted-write completion on the sender -- the note says so).

A pair that cannot be measured (NCCL missing, init timeout, child died, no rate) is a pair WITHOUT number whose note is the reason;
nothing here raises into the probe.  Each pair-run has its own wall cap (the parent kills only its own children).

STDLIB ONLY at module level, like ``bar1_probe`` whose process plumbing (``_spawn_all``) it shares.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.rigmon.bar1_probe import _free_port, _median, _short, _spawn_all

__all__ = [
    "NCCL_PAIR",
    "MARKER",
    "DEFAULT_TIMEOUT_S",
    "NcclResult",
    "rank_command",
    "rank_env",
    "parse_transports",
    "merge_pair",
    "run_nccl_probe",
    "run_rank",
]

#: Transport label base of every pair of this probe; the chosen NCCL transport is appended in brackets.
NCCL_PAIR = "nccl send/recv"

MARKER = "NCCLPROBE "

#: Whole NCCL step (all pair-runs).  BAR1 + NCCL together stay under 8 minutes (order 1006): 240 s each.
DEFAULT_TIMEOUT_S = 240.0

_RATE_BYTES = 64 * 1024 * 1024
_RATE_TRANSFERS = 4
_RATE_REPEATS = 5
_LAT_BYTES = 4096
_LAT_ITERS = 200
_LAT_WARMUP = 20
_DEV_ROUNDS = 200

_VIA = re.compile(r"\bvia\s+([A-Za-z0-9_/\.\-]+)")


class NcclResult:
    def __init__(self, pairs: Optional[List[dict]] = None, reason: str = "", seconds: Optional[float] = None,
                 transports: Optional[List[str]] = None):
        self.pairs = pairs or []
        self.reason = reason
        self.seconds = seconds
        self.transports = transports or []

    def to_json(self) -> dict:
        return {"pairs": self.pairs, "reason": self.reason, "seconds": self.seconds, "transports": list(self.transports)}


# ---------------------------------------------------------------------------
# the parent
# ---------------------------------------------------------------------------


def rank_command(python: str, rank: int, uuids: Sequence[str], timeout_s: float) -> List[str]:
    """Command of rank ``rank`` of one pair-run; ``uuids`` are the two cards in rank order."""
    return [python, "-m", "sglang.srt.rigmon.nccl_probe", "--rank", str(rank), "--uuids", ",".join(uuids),
            "--timeout-s", str(int(timeout_s * 0.8))]


def rank_env(uuids: Sequence[str], port: int, base: Optional[dict] = None) -> dict:
    """Environment of a pair-run child: the caller's own (NCCL as in operation: nothing disabled, nothing forced), the two
    cards by UUID in PCI order, a local rendezvous, and ``NCCL_DEBUG=INFO`` -- for THIS child only, to read the transport."""
    import os

    env = dict(base if base is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    env["NCCL_DEBUG"] = "INFO"
    env.pop("NCCL_DEBUG_FILE", None)       # the log must reach the child's stderr, where the parent reads it
    return env


def parse_transports(stderr_text: str) -> List[str]:
    """The transports NCCL named ("via P2P/CUMEM", "via SHM/direct/direct", "via NET/Socket/0", ...), first-seen order."""
    seen: List[str] = []
    for line in (stderr_text or "").splitlines():
        if "Channel" in line or "via" in line:
            for m in _VIA.finditer(line):
                t = m.group(1)
                if t not in seen:
                    seen.append(t)
    return seen


def parse_rank_output(out: str) -> Optional[dict]:
    rep = None
    for line in (out or "").splitlines():
        if line.startswith(MARKER):
            try:
                rep = json.loads(line[len(MARKER):])
            except ValueError:
                rep = None
    return rep


def merge_pair(
    uuids: Sequence[str],
    results: Sequence[Tuple[int, str, str]],
) -> Tuple[List[dict], List[str]]:
    """Merge the two children of one pair-run into the two directed pairs (rank 0 -> 1, rank 1 -> 0) and the transports seen.

    A direction is measured only when the child that OWNS it reported a rate for exactly that direction AND sat on the card it
    was assigned; otherwise the pair has no number and its note is the reason."""
    transports: List[str] = []
    for _rc, _out, err in results:
        for t in parse_transports(err):
            if t not in transports:
                transports.append(t)
    reps: List[Optional[dict]] = []
    why: Dict[int, str] = {}
    for r in (0, 1):
        rc, out, err = results[r] if r < len(results) else (-1, "", "child did not start")
        rep = parse_rank_output(out)
        if rep is None:
            tail = _short((err or "").strip().splitlines()[-1] if (err or "").strip() else "no output")
            why[r] = f"rank {r} ({uuids[r]}) printed no report (rc={rc}): {tail}"
        elif rep.get("uuid") != uuids[r]:
            why[r] = f"rank {r} was assigned {uuids[r]} but reported card {rep.get('uuid')}: its numbers are discarded"
            rep = None
        elif rep.get("failed"):
            f = rep["failed"]
            why[r] = f"NCCL did not come up on rank {r} (stage {f.get('stage')}): {_short(f.get('reason', ''))}"
            rep = None
        reps.append(rep)
    via = "/".join(transports) if transports else "transport not readable from the NCCL log"
    pairs: List[dict] = []
    for s, d in ((0, 1), (1, 0)):
        # the RECEIVER measured the rate of s -> d; the SENDER started the ping-pong of s -> d
        rate_row = next((x for x in (reps[d] or {}).get("rate") or [] if x.get("src") == s and x.get("dst") == d), None)
        lat_row = next((x for x in (reps[s] or {}).get("lat") or [] if x.get("src") == s and x.get("dst") == d), None)
        bw = rate_row.get("gbs") if rate_row else None
        ok = isinstance(bw, (int, float)) and not isinstance(bw, bool) and bw > 0
        if ok:
            lat = lat_row.get("latency_us") if lat_row else None
            lat_dev = lat_row.get("latency_device_us") if lat_row else None
            note = (f"NCCL chose: {via}. rate = bytes received / receiver wall time, {_RATE_TRANSFERS} x {_RATE_BYTES >> 20} MiB "
                    f"send/recv, median of {_RATE_REPEATS}; latency = 4 kB ping-pong round trip / 2 (delivery incl. the "
                    f"receiver's turn-around), median of {lat_row.get('n', '?') if lat_row else '?'}")
        else:
            lat = None
            lat_dev = None
            # the receiver owns the rate: its failure is the reason; else the sender's; else what is known
            reason = why.get(d) or why.get(s) or "the receiver reported no rate for this direction"
            note = f"NCCL {s}->{d} not measured: {reason}"
        pairs.append({"src_uuid": uuids[s], "dst_uuid": uuids[d], "bandwidth_gbs": bw if ok else None,
                      "latency_us": lat if ok else None,
                      "latency_device_us": lat_dev if ok else None,
                      "latency_device_kind": ("4-kB-Ping-Pong, 200 Runden hintereinander im Stream, ein Synchronize am Ende, Rundlauf / 2")
                      if ok and lat_dev is not None else "",
                      "transport": f"{NCCL_PAIR} ({via})",
                      "peer_access": False, "bytes_moved": _RATE_BYTES if ok else 0, "note": note})
    return pairs, transports


def run_nccl_probe(
    gpus: Sequence[dict],
    *,
    python: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: Optional[Callable[[Sequence[List[str]], dict, float], Sequence[Tuple[int, str, str]]]] = None,
    env: Optional[dict] = None,
    port: Optional[int] = None,
) -> NcclResult:
    """NCCL send/recv for every ordered pair of ``gpus`` (dicts with ``uuid``).  Never raises for a failed measurement.

    One pair-run per UNORDERED pair (both directions inside it), each with its own wall cap = ``timeout_s`` / pair count
    (at least 30 s); a pair-run that hangs ends at its cap and the next one starts."""
    uuids = [str(g["uuid"]) for g in gpus]
    if len(uuids) < 2:
        return NcclResult(reason="fewer than two cards: no pair, no NCCL send/recv")
    t0 = time.time()
    combos = [(i, j) for i in range(len(uuids)) for j in range(i + 1, len(uuids))]
    cap = max(30.0, timeout_s / len(combos))
    all_pairs: List[dict] = []
    transports: List[str] = []
    for (i, j) in combos:
        pair_uuids = [uuids[i], uuids[j]]
        cmds = [rank_command(python or sys.executable, r, pair_uuids, cap) for r in (0, 1)]
        e = rank_env(pair_uuids, port if port is not None else _free_port(), base=env)
        left = timeout_s - (time.time() - t0)
        try:
            if left < 10.0:
                raise TimeoutError(f"the NCCL step's own cap of {timeout_s:.0f} s is used up before this pair")
            results = list((runner or _spawn_all)(cmds, e, min(cap, left)))
        except Exception as ex:  # noqa: BLE001 -- a spawn failure / used-up budget is a reason, not a crash
            results = [(-1, "", f"{type(ex).__name__}: {ex}")] * 2
        pairs, tr = merge_pair(pair_uuids, results)
        all_pairs.extend(pairs)
        for t in tr:
            if t not in transports:
                transports.append(t)
    all_pairs.sort(key=lambda p: (uuids.index(p["src_uuid"]), uuids.index(p["dst_uuid"])))
    missing = [p for p in all_pairs if p["bandwidth_gbs"] is None]
    reason = ""
    if missing:
        distinct: List[str] = []
        for p in missing:
            if p["note"] not in distinct:
                distinct.append(p["note"])
        reason = "; ".join(distinct[:3]) + (f" (+{len(distinct) - 3} more)" if len(distinct) > 3 else "")
    return NcclResult(pairs=all_pairs, reason=reason, seconds=round(time.time() - t0, 1), transports=transports)


# ---------------------------------------------------------------------------
# the rank worker (the child)
# ---------------------------------------------------------------------------


def run_rank(rank: int, uuids: Sequence[str], timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Report of rank ``rank`` of a pair-run (never raises: a failure is ``{"failed": {stage, reason}}``)."""
    from datetime import timedelta

    me = uuids[rank]
    peer = 1 - rank
    rep: Dict[str, object] = {"rank": rank, "uuid": me}
    try:
        import torch
        import torch.distributed as dist

        from sglang.srt.rigmon.card_probe import _inventory

        gpus, _driver = _inventory()
        mine = [g for g in gpus if g["uuid"] == me]
        if not mine:
            rep["failed"] = {"stage": "identity", "reason": f"card {me} is not visible to this process "
                             f"(visible: {[g['uuid'] for g in gpus]})"}
            return rep
        idx = int(mine[0]["cuda_index"])
        torch.cuda.set_device(idx)
        dev = torch.device("cuda", idx)
        dist.init_process_group(backend="nccl", rank=rank, world_size=2, device_id=dev,
                                timeout=timedelta(seconds=timeout_s))
        warm = torch.ones(1, device=dev)
        dist.all_reduce(warm)                       # builds the communicator; the transport lines appear here / at the first send
        torch.cuda.synchronize(dev)
        rate: List[dict] = []
        lat: List[dict] = []
        big = torch.empty(_RATE_BYTES, dtype=torch.uint8, device=dev)
        small = torch.empty(_LAT_BYTES, dtype=torch.uint8, device=dev)
        for s in (0, 1):                            # direction s -> 1-s, both inside this pair-run
            d = 1 - s
            # --- rate (the receiver times it)
            samples: List[float] = []
            for rep_i in range(_RATE_REPEATS + 1):  # the first repeat is the warm-up (lazy p2p connection)
                dist.barrier()
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                for _ in range(_RATE_TRANSFERS):
                    if rank == s:
                        dist.send(big, dst=d)
                    else:
                        dist.recv(big, src=s)
                torch.cuda.synchronize(dev)
                dt = time.perf_counter() - t0
                if rank == d and rep_i > 0 and dt > 0:
                    samples.append(_RATE_TRANSFERS * _RATE_BYTES / 1e9 / dt)
            if rank == d:
                rate.append({"src": s, "dst": d, "gbs": round(_median(samples), 3) if samples else None,
                             "samples_gbs": [round(x, 3) for x in samples]})
            # --- latency (the sender starts the ping-pong; round trip / 2)
            dist.barrier()
            rtts: List[float] = []
            for it in range(_LAT_WARMUP + _LAT_ITERS):
                if rank == s:
                    t0 = time.perf_counter()
                    dist.send(small, dst=d)
                    dist.recv(small, src=d)
                    torch.cuda.synchronize(dev)
                    if it >= _LAT_WARMUP:
                        rtts.append((time.perf_counter() - t0) * 1e6 / 2.0)
                else:
                    dist.recv(small, src=s)
                    dist.send(small, dst=s)
                    torch.cuda.synchronize(dev)
            # SECOND number: the same ping-pong, ``_DEV_ROUNDS`` rounds enqueued back to back on the stream with ONE host
            # synchronisation at the end: total / (2 x rounds) = one-way time without the per-round launch + sync floor.
            dist.barrier()
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(_DEV_ROUNDS):
                if rank == s:
                    dist.send(small, dst=d)
                    dist.recv(small, src=d)
                else:
                    dist.recv(small, src=s)
                    dist.send(small, dst=s)
            torch.cuda.synchronize(dev)
            dev_us = (time.perf_counter() - t0) * 1e6 / (2.0 * _DEV_ROUNDS)
            if rank == s:
                lat.append({"src": s, "dst": d, "latency_us": round(_median(rtts), 1), "min_us": round(min(rtts), 1), "n": len(rtts),
                            "latency_device_us": round(dev_us, 2), "device_rounds": _DEV_ROUNDS})
        rep["rate"] = rate
        rep["lat"] = lat
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        return rep
    except Exception as ex:  # noqa: BLE001
        rep["failed"] = {"stage": "worker", "reason": f"{type(ex).__name__}: {ex}"}
        return rep


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sglang.srt.rigmon.nccl_probe",
                                 description="NCCL send/recv per ordered card pair (parent: --cards; child: --rank)")
    ap.add_argument("--rank", type=int, default=-1, help="internal: run as the child of this rank (pair-run)")
    ap.add_argument("--uuids", default="", help="child: the two card UUIDs in rank order")
    ap.add_argument("--cards", default="", help="parent: NVML indexes, e.g. 0,1,2")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.rank >= 0:
        uuids = [u for u in a.uuids.split(",") if u]
        print(MARKER + json.dumps(run_rank(a.rank, uuids, a.timeout_s)), flush=True)
        return 0
    from sglang.srt.rigmon.hardware_profile import read_nvml

    cards = {c["nvml_index"]: c for c in read_nvml()[0]}
    idx = [int(x) for x in a.cards.split(",") if x.strip()]
    if len(idx) < 2 or any(i not in cards for i in idx):
        print(f"--cards needs at least two NVML indexes known to NVML ({sorted(cards)})", file=sys.stderr)
        return 2
    res = run_nccl_probe([{"uuid": cards[i]["uuid"]} for i in idx], timeout_s=a.timeout_s)
    names = {c["uuid"]: f"{c['nvml_index']}:{c['name']}" for c in cards.values()}
    if a.json:
        print(json.dumps(res.to_json(), indent=1))
    else:
        for p in res.pairs:
            v = p["bandwidth_gbs"]
            print(f"{names[p['src_uuid']]:30s} -> {names[p['dst_uuid']]:30s} "
                  + (f"{v:8.3f} GB/s {p['latency_us']:8.1f} us  [{p['transport']}]" if v is not None else "nicht gemessen: " + p["note"]))
        if res.reason:
            print("REASON:", res.reason)
    return 0 if not res.reason else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
