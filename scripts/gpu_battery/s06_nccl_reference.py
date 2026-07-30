#!/usr/bin/env python3
"""NCCL / system-RAM reference measurement in the #279 format.

The HTCCL path dispatcher's third rate source does not exist yet; its FORMAT
does (htccl_path_rates.new_nccl_reference_envelope, schema_version 1). This
script produces exactly that, so the result is loadable by load_nccl_reference
without a single line of glue.

WHAT IS MEASURED, AND WHY IN THIS SHAPE

* Per row BOTH p50 and p99, in every arm. The #278 wrap-up flagged that its
  load axis had been taken asymmetrically -- p50 on one side, p99 on the other
  -- which made the two sides incomparable and the load verdict unusable. The
  schema makes both mandatory; this script fills both mandatory fields in both
  arms rather than filling p99 only where a tail was expected.

* An explicit LOAD arm next to the idle arm, over the same pairs, sizes and
  iteration counts. Idle rows feed the cost model (p50), load rows feed the
  pressure view (p99). The load is a continuous pinned-host <-> device stream
  on both participating cards: the PCIe bus is the contended resource on this
  rig, and a load arm that does not contend for it measures nothing.

* DIRECTED send/recv, not only symmetric all_reduce. The rig is asymmetric by
  construction (full-BAR 5090 against two windowed 3080s) and Q6 of the
  re-probe asks whether a broadcast topology can exploit that. A symmetric
  collective averages the asymmetry away.

TIMING CONVENTION, stated because it is not self-evident:

* all_reduce rows carry the duration of the collective itself
  (timing="op_duration"), measured with CUDA events on rank 0.
* send_recv rows carry a DIRECTED payload transfer followed by a 4-byte ack
  (timing="directed_ack_round_trip"), timed on the sender. The constant ack
  overhead is also recorded per row as baseline_ack_us. The consumer fits an
  affine model base + per_byte*bytes, so a constant lands in `base` where it
  belongs and does not distort the per-byte term. Halving a symmetric
  ping-pong instead would have averaged exactly the asymmetry we came for.

Cards are named by PCI bus id everywhere. Bare indices are not quotable on
this rig: torch's order and NVML's order differ and the mapping shifts.

RANK PINNING -- the 2026-07-30 hang

The first version launched both ranks with CUDA_VISIBLE_DEVICES=<both cards of
the pair> and had each rank use cuda:0, on the assumption that the environment
variable already pinned the process to one card. It does not: both processes
saw both cards, so both payload tensors landed on the FIRST card of the pair.
The first collective in the loop is dist.barrier(), which -- with no bound
device and no device used yet -- falls back to `rank % visible_device_count`,
i.e. rank 0 on card 0 and rank 1 on card 1. The barrier therefore built a
working communicator on two DIFFERENT devices, and the all_reduce that followed
asked for a communicator on the SAME device from both ranks. Rank 0 already had
one from the barrier and enqueued straight into it; rank 1 had to create a new
one and blocked in the rendezvous for a peer that never came. That is exactly
the pair of stacks the run recorded: rank 0 in cuda.synchronize() after
all_reduce, rank 1 inside all_reduce. The s01 producer had the same pinning bug
and failed loudly instead ("Duplicate GPU detected") because it has no barrier
in front of its first collective to seed the mismatched communicator.

The fix is process-level isolation: each rank gets CUDA_VISIBLE_DEVICES with
EXACTLY ONE card, so cuda:0 is unambiguous in-process and the barrier's
`rank % 1` fallback cannot diverge from it. On top of that the rank checks its
own card's PCI bus id against the one the parent assigned, and the two ranks
exchange those ids over the store before the first collective -- two ranks on
one card is then a named error in the first second, not a wedge.

Usage:
    python s06_nccl_reference.py --out <dir>/nccl_reference.json [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

KIB = 1024
MIB = 1024 * 1024

# Kept short on purpose: the whole step has a ~15 minute budget and a ladder
# that cannot be finished is worth less than a shorter one that can. Four
# decades are enough to fit base + per_byte.
SIZE_LADDER = (64 * KIB, 1 * MIB, 8 * MIB, 64 * MIB)
ITERS = {64 * KIB: 60, 1 * MIB: 50, 8 * MIB: 30, 64 * MIB: 15}
WARMUP = 5

IDLE_ARM = "idle"
LOAD_ARM = "host_stream_64mib"

WORKER_FLAG = "--worker-rank"

# The worker inherits the shortened ladder through the environment rather than
# through argv, so the two ranks cannot be started with different plans.
ENV_SIZES = "BATTERY_S06_SIZES"
ENV_ARMS = "BATTERY_S06_ARMS"
ENV_PG_TIMEOUT = "BATTERY_S06_PG_TIMEOUT_S"

# Worker exit codes that name a precondition rather than a measurement failure.
RC_DEVICE_COUNT = 5
RC_DEVICE_IDENTITY = 6
RC_SAME_CARD = 7
RC_LOCAL_PROBE = 8


class PreconditionFailed(Exception):
    """A rank-local precondition that must end the process, not a collective."""

    def __init__(self, rc: int, message: str):
        super().__init__(message)
        self.rc = rc


def iters_for(nbytes: int) -> int:
    if nbytes in ITERS:
        return ITERS[nbytes]
    if nbytes <= 1 * MIB:
        return 50
    if nbytes <= 8 * MIB:
        return 30
    return 15


def parse_sizes(text: str):
    sizes = tuple(int(x) for x in text.replace(" ", "").split(",") if x)
    if not sizes:
        raise ValueError("empty size ladder")
    for s in sizes:
        if s < 4 or s % 4:
            raise ValueError(f"size {s}: not a multiple of 4 bytes")
    return sizes


def parse_arms(text: str):
    arms = tuple(x for x in text.replace(" ", "").split(",") if x)
    unknown = [a for a in arms if a not in (IDLE_ARM, LOAD_ARM)]
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    if not arms:
        raise ValueError("no arm selected")
    return arms


# ---------------------------------------------------------------------------
# statistics without a numpy dependency
# ---------------------------------------------------------------------------


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


# ---------------------------------------------------------------------------
# progress breadcrumb: what was the rank doing when it stopped moving
# ---------------------------------------------------------------------------


def note_progress(path: str, text: str) -> None:
    """Overwrite the breadcrumb and get it onto the disk immediately.

    A breadcrumb that is still in a buffer when the process is killed answers
    nothing, so this pays for an fsync per measurement -- a few dozen per run.
    """
    try:
        with open(path, "w") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {text}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def read_progress(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "no progress recorded"


def p2p_readiness_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "p2p_readiness"
    )


# ---------------------------------------------------------------------------
# worker: runs inside a subprocess pinned to exactly ONE card
# ---------------------------------------------------------------------------


def _host_load(stop_flag, device):
    """A continuous pinned-host <-> device stream on a side stream.

    This is the foreign load. It runs on its own stream so it contends for the
    bus and the copy engines without serialising against the collective's
    stream, which is what a real neighbour process does.
    """
    import torch

    stream = torch.cuda.Stream(device=device)
    host = torch.empty(64 * MIB, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(64 * MIB, dtype=torch.uint8, device=device)
    with torch.cuda.stream(stream):
        while not stop_flag["stop"]:
            dev.copy_(host, non_blocking=True)
            host.copy_(dev, non_blocking=True)
            stream.synchronize()


def _rank_local_checks(rank: int, progress: str) -> str:
    """Everything that can be decided WITHOUT the peer, decided first.

    Standing rule after four sightings of the same hang family: the local
    precondition is tested before the group collective. A card that is not
    there, not usable, or not the card this rank was assigned must end the
    process with a named error -- inside a collective the same fact is a wedge
    that costs the whole step's budget.

    Returns the rank's own PCI bus id.
    """
    import torch

    sys.path.insert(0, p2p_readiness_path())
    from p2p_common import cuda_pci_bus_id

    note_progress(progress, "rank-local: visibility")
    visible = torch.cuda.device_count()
    if visible != 1:
        raise PreconditionFailed(
            RC_DEVICE_COUNT,
            # "erwartet genau 1" is asserted on by
            # test_gpu_battery_checks.py; the marker stays, the rest does not.
            f"rank {rank}: {visible} cards visible, erwartet genau 1 "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
            "Two visible cards are the 2026-07-30 hang: the barrier falls "
            "back to rank modulo 2, the payload sits on cuda:0.",
        )

    torch.cuda.set_device(0)

    note_progress(progress, "rank-local: CUDA probe")
    probe = torch.ones(4096, dtype=torch.float32, device="cuda:0")
    probe.mul_(2.0)
    total = float(probe.sum().item())
    torch.cuda.synchronize()
    del probe
    torch.cuda.empty_cache()
    if total != 8192.0:
        raise PreconditionFailed(
            RC_LOCAL_PROBE,
            f"rank {rank}: rank-local CUDA probe returns {total}, expected 8192.0",
        )

    # "Kartenidentitaet" is asserted on by test_gpu_battery_checks.py.
    note_progress(progress, "rank-local: Kartenidentitaet")
    own = cuda_pci_bus_id(0)
    expected = os.environ[f"BATTERY_PCI_{rank}"]
    if own != expected:
        raise PreconditionFailed(
            RC_DEVICE_IDENTITY,
            f"rank {rank}: cuda:0 points at {own}, the parent assigned "
            f"{expected} -- CUDA order and NVML order differ on this rig, the "
            "assignment did not arrive",
        )
    return own


def worker(rank: int, out_path: str) -> int:
    """Named preconditions leave through a documented exit code, not a stack."""
    try:
        return _worker(rank, out_path)
    except PreconditionFailed as exc:
        note_progress(f"{out_path}.rank{rank}.progress", f"abort: {exc}")
        print(str(exc), file=sys.stderr, flush=True)
        return exc.rc


def _worker(rank: int, out_path: str) -> int:
    import threading

    import torch
    import torch.distributed as dist

    progress = f"{out_path}.rank{rank}.progress"
    default_sizes = ",".join(str(s) for s in SIZE_LADDER)
    sizes = parse_sizes(os.environ.get(ENV_SIZES) or default_sizes)
    arms = parse_arms(os.environ.get(ENV_ARMS) or ",".join((IDLE_ARM, LOAD_ARM)))
    pg_timeout = datetime.timedelta(seconds=int(os.environ.get(ENV_PG_TIMEOUT) or 300))

    own_pci = _rank_local_checks(rank, progress)

    # Bounded handshake BEFORE the first collective. The store's wait honours a
    # timeout; a NCCL collective against a dead peer does not. This is also
    # where the two ranks compare cards: identical ids mean both ranks were
    # pinned to one card, which NCCL answers with either a duplicate-GPU error
    # or a rendezvous that never completes, depending on what ran before it.
    note_progress(progress, "handshake")
    store = dist.TCPStore(
        os.environ.get("MASTER_ADDR", "127.0.0.1"),
        int(os.environ["MASTER_PORT"]),
        2,
        rank == 0,
        timeout=pg_timeout,
    )
    store.set(f"battery_s06_pci_{rank}", own_pci)
    peer_key = f"battery_s06_pci_{1 - rank}"
    store.wait([peer_key], pg_timeout)
    peer_pci = store.get(peer_key).decode()
    if peer_pci == own_pci:
        raise PreconditionFailed(
            RC_SAME_CARD,
            f"rank {rank}: both ranks on {own_pci} -- one card cannot carry "
            "two ranks of one group",
        )

    note_progress(progress, "init_process_group")
    device = torch.device("cuda:0")
    # device_id binds the group to this card, so barrier() never has to guess a
    # device from rank % visible_count. With one visible card the guess would
    # land right anyway; binding it makes that independent of the environment.
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        store=store,
        timeout=pg_timeout,
        device_id=device,
    )

    rows = []
    stop_flag = {"stop": True}
    load_thread = None

    def start_load():
        nonlocal load_thread
        stop_flag["stop"] = False
        load_thread = threading.Thread(
            target=_host_load, args=(stop_flag, device), daemon=True
        )
        load_thread.start()

    def stop_load():
        nonlocal load_thread
        stop_flag["stop"] = True
        if load_thread is not None:
            load_thread.join(timeout=30)
            load_thread = None

    def timed_all_reduce(nbytes, iters):
        buf = torch.ones(nbytes // 4, dtype=torch.float32, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for i in range(iters + WARMUP):
            dist.barrier()
            start.record()
            dist.all_reduce(buf)
            end.record()
            torch.cuda.synchronize()
            if i >= WARMUP:
                samples.append(start.elapsed_time(end) * 1000.0)  # ms -> us
        del buf
        torch.cuda.empty_cache()
        return samples

    def timed_directed(nbytes, iters, sender: int):
        """One directed payload plus a 4-byte ack, timed on the sender."""
        payload = torch.empty(max(nbytes, 4) // 4, dtype=torch.float32, device=device)
        ack = torch.zeros(1, dtype=torch.float32, device=device)
        peer = 1 - rank
        samples = []
        for i in range(iters + WARMUP):
            dist.barrier()
            t0 = time.perf_counter()
            if rank == sender:
                dist.send(payload, peer)
                dist.recv(ack, peer)
            else:
                dist.recv(payload, peer)
                dist.send(ack, peer)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= WARMUP and rank == sender:
                samples.append((t1 - t0) * 1e6)
        del payload, ack
        torch.cuda.empty_cache()
        return samples

    pci = [os.environ["BATTERY_PCI_0"], os.environ["BATTERY_PCI_1"]]

    for arm in arms:
        if arm == LOAD_ARM:
            start_load()
            time.sleep(2)  # let the stream reach steady state before measuring

        # all_reduce: symmetric, recorded against the sorted pair
        for nbytes in sizes:
            note_progress(progress, f"{arm}/all_reduce/{nbytes}B")
            samples = timed_all_reduce(nbytes, iters_for(nbytes))
            if rank == 0:
                src, dst = sorted(pci)
                rows.append(
                    {
                        "op": "all_reduce",
                        "transport": None,  # filled by the parent from the log
                        "world": 2,
                        "src_pci": src,
                        "dst_pci": dst,
                        "size_bytes": nbytes,
                        "iters": len(samples),
                        "p50_us": round(percentile(samples, 0.50), 3),
                        "p99_us": round(percentile(samples, 0.99), 3),
                        "load": arm,
                        "timing": "op_duration",
                    }
                )

        # send/recv: both directions, so the asymmetry survives
        baseline = {}
        for sender in (0, 1):
            note_progress(progress, f"{arm}/send_recv_baseline/sender{sender}")
            base_samples = timed_directed(4, 30, sender)
            if rank == sender:
                baseline[sender] = round(percentile(base_samples, 0.50), 3)

        for sender in (0, 1):
            for nbytes in sizes:
                note_progress(progress, f"{arm}/send_recv/{nbytes}B/sender{sender}")
                samples = timed_directed(nbytes, iters_for(nbytes), sender)
                if rank == sender:
                    rows.append(
                        {
                            "op": "send_recv",
                            "transport": None,
                            "world": 2,
                            "src_pci": pci[sender],
                            "dst_pci": pci[1 - sender],
                            "size_bytes": nbytes,
                            "iters": len(samples),
                            "p50_us": round(percentile(samples, 0.50), 3),
                            "p99_us": round(percentile(samples, 0.99), 3),
                            "load": arm,
                            "timing": "directed_ack_round_trip",
                            "baseline_ack_us": baseline.get(sender),
                        }
                    )

        if arm == LOAD_ARM:
            stop_load()

    # Rank 1 owns the rows it timed (the sender side of its direction), so both
    # ranks write and the parent merges. A single writer would silently drop
    # one direction.
    note_progress(progress, "writing the fragment")
    with open(f"{out_path}.rank{rank}", "w") as f:
        json.dump(rows, f)

    dist.barrier()
    dist.destroy_process_group()
    note_progress(progress, "done")
    return 0


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def register_pid(pid: int) -> None:
    """Hand the pid to run_step.sh so its outer timeout dumps it.

    The 2026-07-30 STOP line reported zero dumped processes: the step never told
    the runner which processes it owned, so the stacks had to be taken by hand.
    """
    pids_file = os.path.join(os.environ.get("BATTERY_STEP_DIR", ""), "pids")
    if not os.environ.get("BATTERY_STEP_DIR"):
        return
    try:
        with open(pids_file, "a") as f:
            f.write(f"{pid}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def pyspy_dump(pid: int, path: str) -> None:
    """Stacks BEFORE the kill, always. A dumped hang does not need a repro."""
    venv_exe = os.path.join(os.path.dirname(sys.executable), "py-spy")
    exe = shutil.which("py-spy") or (venv_exe if os.path.exists(venv_exe) else None)
    if exe is None:
        with open(path, "w") as f:
            f.write("py-spy not found -- no dump possible\n")
        return
    try:
        with open(path, "w") as f:
            subprocess.run(
                [exe, "dump", "--pid", str(pid)],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        with open(path, "a") as f:
            f.write(f"py-spy failed: {exc}\n")


def worker_argv(rank: int, frag_path: str):
    """The command line of one rank. A seam, so the timeout and logging paths
    are falsifiable with a stand-in child on a machine without a card."""
    return [
        sys.executable,
        os.path.abspath(__file__),
        WORKER_FLAG,
        str(rank),
        "--frag",
        frag_path,
    ]


def terminate_own_group(proc) -> None:
    """Kill exactly the process group this parent created, nothing wider.

    Every child is started with start_new_session=True, so its process group id
    equals its pid and the group contains only its own descendants. The pgid is
    verified against the pid before killpg runs: a signal to a group we did not
    create would reach foreign server processes on this shared box.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    try:
        if pgid == proc.pid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        if pgid == proc.pid:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_pair(cuda_ids, pci_ids, frag_path, timeout_s, log_prefix, sizes, arms):
    """Two pinned single-card processes, one shared deadline.

    Each rank sees EXACTLY ONE card. The pair's two processes get their own
    session, so a timeout can be answered with killpg on a group this parent
    created and on nothing else.

    stdout/stderr of each rank go straight into a file handle. A PIPE with no
    concurrent reader fills its 64 KiB kernel buffer and blocks the child
    inside NCCL_DEBUG=INFO output -- the previous version read the pipes one
    rank after the other, so rank 1 could be blocked on a full pipe for as long
    as rank 0 was being waited on.
    """
    port = free_port()
    base_env = dict(os.environ)
    base_env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NCCL_DEBUG": "INFO",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "BATTERY_PCI_0": pci_ids[0],
            "BATTERY_PCI_1": pci_ids[1],
            ENV_SIZES: ",".join(str(s) for s in sizes),
            ENV_ARMS: ",".join(arms),
            # The group's own rendezvous must give up well inside the pair
            # deadline, so a wedged peer produces a NCCL error with a stack
            # rather than a process the parent has to kill.
            ENV_PG_TIMEOUT: str(max(60, timeout_s // 3)),
        }
    )

    procs, log_paths = [], []
    for rank in (0, 1):
        env = dict(base_env)
        # One physical card per process. Inside the child cuda:0 is then
        # unambiguous and no in-process logical/physical mapping is needed.
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_ids[rank])
        log_path = f"{log_prefix}.rank{rank}.log"
        log_paths.append(log_path)
        with open(log_path, "w") as fh:
            proc = subprocess.Popen(
                worker_argv(rank, frag_path),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        procs.append(proc)
        register_pid(proc.pid)

    deadline = time.monotonic() + timeout_s
    status, detail = "ok", ""
    timed_out = []
    for rank, proc in enumerate(procs):
        remaining = deadline - time.monotonic()
        try:
            rc = proc.wait(timeout=max(remaining, 0.0))
        except subprocess.TimeoutExpired:
            timed_out.append(rank)
            continue
        if rc != 0:
            status = f"rank {rank} exited {rc}"

    if timed_out:
        where = []
        for rank in timed_out:
            proc = procs[rank]
            pyspy_dump(proc.pid, f"{log_prefix}.rank{rank}.pyspy.txt")
            step = read_progress(f"{frag_path}.rank{rank}.progress")
            # "Rang <n>" is asserted on by test_gpu_battery_checks.py.
            where.append(f"Rang {rank} at {step}")
        for proc in procs:
            terminate_own_group(proc)
        # The exact wording is asserted on by test_gpu_battery_checks.py.
        status = f"timeout nach {timeout_s}s"
        detail = "; ".join(where)

    logs = []
    for path in log_paths:
        try:
            with open(path, errors="replace") as f:
                logs.append(f.read())
        except OSError as exc:
            logs.append(f"[log {path} not readable: {exc}]")
    for rank in (0, 1):
        try:
            os.unlink(f"{frag_path}.rank{rank}.progress")
        except OSError:
            pass
    return status, detail, "\n".join(logs)


def write_payload(payload, all_rows, out_path) -> None:
    """Written after EVERY pair, not only at the end.

    An aborted step still leaves a file whose pairs_status names what happened
    to each pair. The check rejects any pair whose status is not "ok", so the
    partial file is a FAIL with a reason instead of a missing artefact.
    """
    payload["rows"] = all_rows
    tmp = f"{out_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(WORKER_FLAG, type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--frag", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="nccl_reference.json")
    ap.add_argument(
        "--log", default=None, help="NCCL_DEBUG output goes here, not to stdout"
    )
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument(
        "--sizes",
        default=None,
        help="comma-separated byte sizes instead of the full ladder "
        "(short evidence run: --sizes 65536,1048576)",
    )
    ap.add_argument(
        "--arms",
        default=None,
        help=f"comma-separated arms instead of both ({IDLE_ARM},{LOAD_ARM})",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.worker_rank is not None:
        return worker(args.worker_rank, args.frag)

    sizes = parse_sizes(args.sizes) if args.sizes else SIZE_LADDER
    arms = parse_arms(args.arms) if args.arms else (IDLE_ARM, LOAD_ARM)

    if args.dry_run:
        print(
            "Plan: per unordered card pair, two subprocesses with EXACTLY "
            "ONE visible card each (CUDA_VISIBLE_DEVICES=<one card>, "
            f"NCCL_DEBUG=INFO); per arm ({', '.join(arms)}) all_reduce over "
            f"the ladder {[s // KIB for s in sizes]} KiB plus send_recv in "
            "BOTH directions; p50 AND p99 per row; output in the "
            "nccl_reference envelope (schema_version 1)."
        )
        return 0

    sys.path.insert(0, p2p_readiness_path())
    from p2p_common import cuda_device_count, cuda_pci_bus_id  # noqa: E402

    repo_python = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
    )
    sys.path.insert(0, repo_python)
    from sglang.srt.distributed.device_communicators.htccl_path_rates import (  # noqa: E402
        new_nccl_reference_envelope,
    )
    from p2p_common import parse_nccl_transports, summarize_transport_classes  # noqa: E402

    n = cuda_device_count()
    if n < 2:
        print(f"need >= 2 cards, found {n}", file=sys.stderr)
        return 4
    idx_pci = {i: cuda_pci_bus_id(i) for i in range(n)}

    payload = new_nccl_reference_envelope()
    payload["host"] = os.uname().nodename
    payload["timing_conventions"] = {
        "op_duration": "duration of the collective, CUDA events on rank 0",
        "directed_ack_round_trip": (
            "directed payload plus a 4-byte ack, timed on the sender; the "
            "constant ack share is recorded per row in baseline_ack_us and "
            "lands in the base term of the affine fit"
        ),
    }
    payload["load_arms"] = {
        IDLE_ARM: "no foreign load",
        LOAD_ARM: "continuous 64 MiB pinned-host <-> device stream on both cards",
    }
    payload["pairs_status"] = []

    log_path = args.log or (os.path.splitext(args.out)[0] + ".nccl.log")
    all_rows = []
    failed = False

    with open(log_path, "w") as logf:
        for a, b in itertools.combinations(range(n), 2):
            frag = f"{args.out}.frag.{a}-{b}"
            # A fresh log segment per pair: the rank logs go to their own files
            # first and are folded in here, so nothing of pair N-1 can be
            # attributed to pair N.
            status, detail, log = run_pair(
                (a, b),
                (idx_pci[a], idx_pci[b]),
                frag,
                args.timeout,
                f"{os.path.splitext(log_path)[0]}.pair-{a}-{b}",
                sizes,
                arms,
            )
            logf.write(f"===== pair cuda:{a} <-> cuda:{b} ({status}) =====\n{log}\n")
            logf.flush()
            os.fsync(logf.fileno())
            transports = summarize_transport_classes(parse_nccl_transports(log))
            # summarize_transport_classes returns {"0->1": "SHM", ...}: the KEY
            # is a direction, the VALUE is the transport class the row field is
            # defined as ("P2P"|"SHM"|"NET"). Taking the key put "0->1" into
            # every row's transport. Two different classes over the two
            # directions is a finding, so both are named rather than one picked.
            classes = sorted(set(transports.values()))
            chosen = "+".join(classes) if classes else None

            rows = []
            for rank in (0, 1):
                part = f"{frag}.rank{rank}"
                if os.path.exists(part):
                    with open(part) as f:
                        rows.extend(json.load(f))
                    os.unlink(part)
            for row in rows:
                row["transport"] = chosen
            all_rows.extend(rows)

            entry = {
                "pci_pair": [idx_pci[a], idx_pci[b]],
                "status": status,
                "transport_summary": transports,
                "rows": len(rows),
            }
            if detail:
                entry["detail"] = detail
            payload["pairs_status"].append(entry)
            if status != "ok":
                failed = True
            # Partial result on disk before the next pair can eat the budget.
            write_payload(payload, all_rows, args.out)
            line = f"{idx_pci[a]} <-> {idx_pci[b]}: {status}, {len(rows)} rows, {transports}"
            if detail:
                line += f" -- {detail}"
            print(line, flush=True)

    write_payload(payload, all_rows, args.out)
    print(f"written: {args.out} ({len(all_rows)} rows); NCCL log: {log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
