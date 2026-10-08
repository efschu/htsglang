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
"""The BAR1 stretch per ORDERED card pair, measured through the real transport.

Order 1006 ("Hardwareprofil messen muss genau diese Werte liefern").  Until this
module the hardware profile said, for every pair, "BAR1 stretch: NOT MEASURED:
no single-process entry that runs it without a server".  That reason was true of
the transport (``barlink_bar1.build_bar1`` needs a ``torch.distributed`` CPU
group with ONE PROCESS PER RANK, the ``dmabuf_holder`` module and the patched
driver's peer-BAR1 reg key) and wrong as a verdict: a measurement run may start
processes.  This module is exactly that entry -- a **parent** that spawns one
child per card and a **rank worker** that is the child:

* each child sees all chosen cards (``CUDA_VISIBLE_DEVICES`` = their UUIDs),
  finds ITS card by NVML UUID (``card_probe._inventory`` -- the #397 identity
  map, never a torch index) and runs on it;
* the children form a gloo group and call ``barlink_bar1.build_bar1`` -- the
  production factory, including the byte-level proof per directed pair;
* then, one directed pair at a time (barrier, sender measures, barrier), the
  sender calls ``transport.pair(dst, nbytes)`` -- the SAME pair sensor the
  barlink matrix planner uses for edge capacities (one-sided posted writes into
  the destination's BAR1 window, rate in GB/s) -- and times 4 kB ``put`` + device
  sync for the latency;
* every child prints ONE ``BAR1PROBE {json}`` line; the parent merges them.

**What a number here means.**  ``bandwidth_gbs`` is the rate of writes from the
source card INTO the destination card's BAR1 aperture (BAR1 is write-posted: the
reverse direction is a different quantity and is measured as its own row; a read
across BAR1 is not offered by the transport).  ``latency_us`` is the median of
``_LAT_ITERS`` 4 kB writes, each followed by a device synchronize.

**A pair that cannot be measured is "nicht gemessen" WITH its reason**, never a
number: the byte proof failed for that direction, the holder module or the reg
key is missing, the extension did not build, a child died or timed out.  The
reason is stored in the probe (``bar1_reason`` / the pair's ``note``) and shown
by the profile.  ``DEFAULT_TIMEOUT_S`` bounds the whole step; the parent kills
only the children it started (by handle, never by pattern).

STDLIB ONLY at module level (the parent runs inside the probe process, the rank
worker imports torch lazily): the merge and the command/env construction are
unit-tested without a GPU.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BAR1_DIRECT",
    "MARKER",
    "DEFAULT_TIMEOUT_S",
    "Bar1Result",
    "rank_command",
    "rank_env",
    "parse_rank_output",
    "merge_reports",
    "run_bar1_probe",
    "run_rank",
]

#: Transport label of every pair of this probe (never mixed with p2p / host staging).
BAR1_DIRECT = "bar1 (direct write into the destination's BAR1)"

#: A child prints exactly one line ``BAR1PROBE <json>``; the parent reads only those.
MARKER = "BAR1PROBE "

#: The whole step (children start, JIT-cached extension loads, byte proofs, all pairs).  A cold extension build
#: is minutes, a warm one seconds; the cap is what the measurement window can afford.
DEFAULT_TIMEOUT_S = 240.0

#: The shared, warm BAR1 extension cache of this rig (the battery's ``BAR1_EXTCACHE`` default).  Used only when
#: it exists and the caller's environment does not name another ``TORCH_EXTENSIONS_DIR``.
DEFAULT_EXTCACHE = "/spinning/barlink_extcache_host"

#: Payload of the bandwidth probe (``transport.pair`` clamps it to the mapped window) and of the latency probe.
_BW_BYTES = 16 * 1024 * 1024
_BW_REPEATS = 3
_LAT_BYTES = 4096
_LAT_ITERS = 200
_LAT_WARMUP = 10
_STREAM_N = 1000
_STREAM_REPEATS = 5


class Bar1Result:
    """What the step produced: ordered pairs (dicts shaped like ``PairMeasurement``), the reason when it did not
    run or only partly ran, and the facts of the run."""

    def __init__(self, pairs: Optional[List[dict]] = None, reason: str = "", seconds: Optional[float] = None,
                 window_mib: Optional[float] = None, notes: Optional[List[str]] = None):
        self.pairs = pairs or []
        self.reason = reason
        self.seconds = seconds
        self.window_mib = window_mib
        self.notes = notes or []

    def to_json(self) -> dict:
        return {"pairs": self.pairs, "reason": self.reason, "seconds": self.seconds, "window_mib": self.window_mib,
                "notes": list(self.notes)}


# ---------------------------------------------------------------------------
# the parent: commands, environment, merge (all without a GPU)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def rank_command(python: str, rank: int, uuids: Sequence[str], timeout_s: float) -> List[str]:
    """The command of rank ``rank``: the card it must run on travels by UUID, in rank order."""
    return [python, "-m", "flliper.srt.rigmon.bar1_probe", "--rank", str(rank), "--uuids", ",".join(uuids),
            # the children's own (gloo) deadline ends BEFORE the parent's kill, so a stuck group reports instead of dying mute
            "--timeout-s", str(int(timeout_s * 0.8))]


def rank_env(uuids: Sequence[str], port: int, base: Optional[dict] = None, extcache: Optional[str] = None) -> dict:
    """Environment of every child.  All chosen cards are visible, by UUID (order of ranks), PCI bus order pinned;
    the rendezvous is local.  ``extcache`` (or the rig's warm cache directory) becomes ``TORCH_EXTENSIONS_DIR``
    unless the caller's environment already names one -- a cold build of the BAR1 extension costs minutes."""
    env = dict(base if base is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    cache = extcache if extcache is not None else (DEFAULT_EXTCACHE if os.path.isdir(DEFAULT_EXTCACHE) else None)
    if cache and not env.get("TORCH_EXTENSIONS_DIR"):
        env["TORCH_EXTENSIONS_DIR"] = cache
    return env


def parse_rank_output(out: str) -> Optional[dict]:
    """The child's report (the last ``BAR1PROBE`` line), or ``None`` when it printed none."""
    rep = None
    for line in (out or "").splitlines():
        if line.startswith(MARKER):
            try:
                rep = json.loads(line[len(MARKER):])
            except ValueError:
                rep = None
    return rep


def _short(s: str, n: int = 300) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def merge_reports(
    uuids: Sequence[str],
    results: Sequence[Tuple[int, str, str]],
) -> Bar1Result:
    """Merge the children's ``(rc, stdout, stderr)`` (rank order) into the step's result.

    Every ordered pair of ``uuids`` appears in the result.  A pair is measured only when the SENDER's child
    reported a rate for exactly that pair AND that child sat on the card it was assigned (its reported UUID is the
    rank's UUID); anything else is a pair without number whose ``note`` is the reason."""
    n = len(uuids)
    reports: List[Optional[dict]] = []
    why_rank: Dict[int, str] = {}
    for r in range(n):
        rc, out, err = results[r] if r < len(results) else (-1, "", "child did not start")
        rep = parse_rank_output(out)
        reports.append(rep)
        if rep is None:
            tail = _short((err or "").strip().splitlines()[-1] if (err or "").strip() else "no output")
            why_rank[r] = f"rank {r} ({uuids[r]}) printed no report (rc={rc}): {tail}"
        elif rep.get("uuid") != uuids[r]:
            why_rank[r] = (f"rank {r} was assigned {uuids[r]} but reported card {rep.get('uuid')}: "
                           "its numbers are discarded (a rate must belong to the card it names)")
            reports[r] = None
        elif rep.get("failed"):
            f = rep["failed"]
            why_rank[r] = f"BAR1 transport did not come up on rank {r} (stage {f.get('stage')}): {_short(f.get('reason', ''))}"
            reports[r] = None
    pairs: List[dict] = []
    window = None
    for s in range(n):
        rows = {}
        if reports[s] is not None:
            for row in reports[s].get("pairs") or []:
                rows[(row.get("src"), row.get("dst"))] = row
            if reports[s].get("window_mib") is not None:
                window = reports[s]["window_mib"] if window is None else min(window, reports[s]["window_mib"])
        for d in range(n):
            if s == d:
                continue
            row = rows.get((s, d))
            bw = row.get("bandwidth_gbs") if row else None
            ok = isinstance(bw, (int, float)) and not isinstance(bw, bool) and bw > 0
            if ok:
                note = (f"one-sided posted writes {row.get('nbytes', 0) >> 20} MiB x{row.get('repeats', '?')} into the "
                        f"destination's BAR1 window ({row.get('window_mib', '?')} MiB mapped); latency = median of "
                        f"{row.get('lat_n', '?')} x 4 kB write+sync, min {row.get('lat_min_us', '?')} us")
                lat = row.get("latency_us")
                lat_dev = row.get("latency_device_us")
            else:
                if row and row.get("reason"):
                    note = f"BAR1 {s}->{d} not measured: {row['reason']}"
                elif s in why_rank:
                    note = why_rank[s]
                elif reports[s] is not None:
                    note = f"BAR1 {s}->{d}: the sender reported no rate for this pair"
                else:
                    note = "BAR1 not measured"
                lat = None
                lat_dev = None
            pairs.append({"src_uuid": uuids[s], "dst_uuid": uuids[d], "bandwidth_gbs": bw if ok else None,
                          "latency_us": lat if ok else None,
                          "latency_device_us": lat_dev if ok else None,
                          "latency_device_kind": ("4-kB-Schreibzugriffe hintereinander im Stream, ein Synchronize am Ende, Zeit je Schreibzugriff "
                                                  "(KEIN Rundlauf)") if ok and lat_dev is not None else "",
                          "transport": BAR1_DIRECT, "peer_access": bool(ok),
                          "bytes_moved": int(row.get("nbytes", 0)) if ok else 0, "note": note})
    missing = [p for p in pairs if p["bandwidth_gbs"] is None]
    if not missing:
        reason = ""
    else:
        distinct = []
        for p in missing:
            if p["note"] not in distinct:
                distinct.append(p["note"])
        reason = "; ".join(distinct[:3]) + (f" (+{len(distinct) - 3} more)" if len(distinct) > 3 else "")
    return Bar1Result(pairs=pairs, reason=reason, window_mib=window)


def _spawn_all(cmds: Sequence[List[str]], env: dict, timeout_s: float) -> List[Tuple[int, str, str]]:
    """Start every rank at once, wait for all with one deadline, kill only OUR children on timeout."""
    procs = [subprocess.Popen(c, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for c in cmds]
    out: List[Tuple[int, str, str]] = []
    deadline = time.time() + timeout_s
    # communicate() of the first child can block while a sibling fills its pipe: drain through threads
    import threading

    bufs: List[Dict[str, Any]] = [{} for _ in procs]

    def drain(i: int):
        try:
            o, e = procs[i].communicate(timeout=max(1.0, deadline - time.time()))
            bufs[i].update(rc=procs[i].returncode, out=o, err=e)
        except subprocess.TimeoutExpired:
            procs[i].kill()
            try:
                o, e = procs[i].communicate(timeout=10)
            except Exception:  # noqa: BLE001
                o, e = "", ""
            bufs[i].update(rc=124, out=o or "", err=(e or "") + f"\ntimeout after {timeout_s:.0f} s")

    th = [threading.Thread(target=drain, args=(i,), daemon=True) for i in range(len(procs))]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=timeout_s + 30)
    for i, b in enumerate(bufs):
        out.append((b.get("rc", -1), b.get("out", ""), b.get("err", "")))
    return out


def run_bar1_probe(
    gpus: Sequence[dict],
    *,
    python: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: Optional[Callable[[Sequence[List[str]], dict, float], Sequence[Tuple[int, str, str]]]] = None,
    env: Optional[dict] = None,
    port: Optional[int] = None,
    extcache: Optional[str] = None,
) -> Bar1Result:
    """Measure the BAR1 stretch of every ordered pair of ``gpus`` (dicts with ``uuid``).

    ``runner(cmds, env, timeout) -> [(rc, stdout, stderr)]`` is injectable for tests.  Never raises for a
    failed measurement: the result then carries pairs without number and the reason."""
    uuids = [str(g["uuid"]) for g in gpus]
    if len(uuids) < 2:
        return Bar1Result(reason="fewer than two cards: no pair, no BAR1 stretch")
    t0 = time.time()
    cmds = [rank_command(python or sys.executable, r, uuids, timeout_s) for r in range(len(uuids))]
    e = rank_env(uuids, port if port is not None else _free_port(), base=env, extcache=extcache)
    try:
        results = list((runner or _spawn_all)(cmds, e, timeout_s))
    except Exception as ex:  # noqa: BLE001 -- a spawn failure is a reason, not a crash of the whole probe
        res = merge_reports(uuids, [])
        res.reason = f"BAR1 children could not be started: {type(ex).__name__}: {ex}"
        res.seconds = round(time.time() - t0, 1)
        return res
    res = merge_reports(uuids, results)
    res.seconds = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------
# the rank worker (the child): torch, gloo, the real transport
# ---------------------------------------------------------------------------


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def run_rank(rank: int, uuids: Sequence[str], timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """The report of rank ``rank`` (never raises: a failure is ``{"failed": {stage, reason}}``)."""
    from datetime import timedelta

    me = uuids[rank]
    rep: Dict[str, Any] = {"rank": rank, "uuid": me}
    try:
        import torch
        import torch.distributed as dist

        from flliper.srt.rigmon.card_probe import _inventory

        gpus, _driver = _inventory()
        mine = [g for g in gpus if g["uuid"] == me]
        if not mine:
            rep["failed"] = {"stage": "identity", "reason": f"card {me} is not visible to this process (visible: "
                             f"{[g['uuid'] for g in gpus]})"}
            return rep
        idx = int(mine[0]["cuda_index"])
        torch.cuda.set_device(idx)
        dev = torch.device("cuda", idx)
        to = timedelta(seconds=timeout_s)
        dist.init_process_group(backend="gloo", rank=rank, world_size=len(uuids), timeout=to)
        group = dist.new_group(list(range(len(uuids))), backend="gloo", timeout=to)
        from flliper.srt.distributed.device_communicators.barlink_bar1 import build_bar1
        from flliper.srt.distributed.device_communicators.barlink_matrix_transport import window_for

        report: dict = {}
        t = build_bar1(group, dev, window_for("", dev), report, group="")
        if t is None:
            rep["failed"] = {"stage": report.get("stage", "unknown"), "reason": report.get("reason", "no reason reported")}
            return rep
        # A transport object with ``report["holds_space"]`` and failed byte proofs exists but handles() is False for
        # everything; ``_measure_pair`` makes the per-pair verdict from the proof flag of each directed pair.
        rep["window_mib"] = round(int(getattr(t, "window_bytes", 0)) / (1 << 20), 2) or None
        rows: List[dict] = []
        for s in range(len(uuids)):
            for d in range(len(uuids)):
                if s == d:
                    continue
                dist.barrier(group)
                if rank == s:
                    try:
                        rows.append(_measure_pair(t, dev, s, d))
                    except Exception as ex:  # noqa: BLE001 -- the barrier sequence must go on: a pair failure is a reason
                        rows.append({"src": s, "dst": d, "reason": f"{type(ex).__name__}: {ex}"})
                dist.barrier(group)
        rep["pairs"] = rows
        rep["bdfs"] = list(getattr(t, "bdfs", []) or [])
        try:
            t.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        return rep
    except Exception as ex:  # noqa: BLE001
        rep["failed"] = {"stage": "worker", "reason": f"{type(ex).__name__}: {ex}"}
        return rep


def _measure_pair(t, dev, s: int, d: int) -> dict:
    """Sender side of one directed pair: rate through the transport's own pair sensor, latency through ``put``."""
    import torch

    row: Dict[str, Any] = {"src": s, "dst": d, "nbytes": _BW_BYTES, "repeats": _BW_REPEATS}
    peer = getattr(t, "_peers", {}).get(d)
    if peer is None or not getattr(peer, "byte_proof", False):
        row["reason"] = ("no byte-level proof for this direction: the destination's window was not mapped or lost bytes "
                         "(the edge is struck regardless of what the driver reports)")
        return row
    row["window_mib"] = round(int(peer.length) / (1 << 20), 2)
    rates = []
    for _ in range(_BW_REPEATS):
        v = t.pair(d, _BW_BYTES)
        if v is None:
            row["reason"] = "transport.pair returned no rate"
            return row
        rates.append(float(v))
    row["nbytes"] = min(_BW_BYTES, int(peer.length))
    row["bandwidth_gbs"] = round(_median(rates), 3)
    row["rate_samples_gbs"] = [round(r, 3) for r in rates]
    src = torch.empty(_LAT_BYTES, dtype=torch.uint8, device=dev)
    for _ in range(_LAT_WARMUP):
        t.put(d, src.data_ptr(), _LAT_BYTES, 0)
    torch.cuda.synchronize(dev)
    lat = []
    for _ in range(_LAT_ITERS):
        t0 = time.perf_counter()
        t.put(d, src.data_ptr(), _LAT_BYTES, 0)
        torch.cuda.synchronize(dev)
        lat.append((time.perf_counter() - t0) * 1e6)
    row["latency_us"] = round(_median(lat), 1)
    row["lat_min_us"] = round(min(lat), 1)
    row["lat_n"] = _LAT_ITERS
    # SECOND number (order 1006, user finding 15:40Z): the figure above carries one kernel launch plus one host synchronisation per
    # write, a floor of the order of 10 us that is NOT the wire latency.  This one issues ``_STREAM_N`` writes back to back on the
    # stream and synchronises ONCE, so the per-write floor is paid once: time / N = the cost of one 4 kB posted write inside a
    # stream.  It is NOT a round trip (a flag round trip needs a kernel that writes a flag and spins on the peer's: not built here).
    per = []
    for _ in range(_STREAM_REPEATS):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(_STREAM_N):
            t.put(d, src.data_ptr(), _LAT_BYTES, 0)
        torch.cuda.synchronize(dev)
        per.append((time.perf_counter() - t0) * 1e6 / _STREAM_N)
    row["latency_device_us"] = round(_median(per), 2)
    row["latency_device_n"] = _STREAM_N * _STREAM_REPEATS
    return row


# ---------------------------------------------------------------------------
# CLI: child (--rank) or parent (--cards)
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m flliper.srt.rigmon.bar1_probe",
                                 description="BAR1 stretch per ordered card pair (parent: --cards; child: --rank)")
    ap.add_argument("--rank", type=int, default=-1, help="internal: run as the child of this rank")
    ap.add_argument("--uuids", default="", help="card UUIDs in rank order (child; parent derives them from --cards)")
    ap.add_argument("--cards", default="", help="parent: NVML indexes, e.g. 0,1,2 (resolved to UUIDs through NVML)")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.rank >= 0:
        uuids = [u for u in a.uuids.split(",") if u]
        print(MARKER + json.dumps(run_rank(a.rank, uuids, a.timeout_s)), flush=True)
        return 0
    from flliper.srt.rigmon.hardware_profile import read_nvml

    cards = {c["nvml_index"]: c for c in read_nvml()[0]}
    idx = [int(x) for x in a.cards.split(",") if x.strip()]
    if len(idx) < 2 or any(i not in cards for i in idx):
        print(f"--cards needs at least two NVML indexes known to NVML ({sorted(cards)})", file=sys.stderr)
        return 2
    res = run_bar1_probe([{"uuid": cards[i]["uuid"]} for i in idx], timeout_s=a.timeout_s)
    names = {c["uuid"]: f"{c['nvml_index']}:{c['name']}" for c in cards.values()}
    if a.json:
        print(json.dumps(res.to_json(), indent=1))
    else:
        for p in res.pairs:
            v = p["bandwidth_gbs"]
            print(f"{names[p['src_uuid']]:30s} -> {names[p['dst_uuid']]:30s} "
                  + (f"{v:8.3f} GB/s {p['latency_us']:8.1f} us" if v is not None else "nicht gemessen: " + p["note"]))
        if res.reason:
            print("REASON:", res.reason)
    return 0 if not res.reason else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
