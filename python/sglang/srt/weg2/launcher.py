"""Weg-2 group launcher (spec section 1.5, record sections 1c-1g).

Two STOCK ``python -m sglang.launch_server`` launches per rig, sequenced so
that exactly one group is awake at the end:

    1. NVML resolve (the 5090 by name, the two 3080s), CUDA_VISIBLE_DEVICES
       pinned by UUID: ordinal 0 = 5090, 1 and 2 = the 3080s.
    2. host ledger (#721) for six processes at BOTH moments -> S, M and the
       tmpfs store size; W20 refuses.
    3. the canonical page store on a RAM-backed tmpfs, sized by the ledger.
    4. group P (PP=3 prefill, :30031) launches with per-card budgets derived
       from NVML total minus the corridor minus the EXPECTED dormant residue
       of group D; R1 on :30031; W7/W10 checked from its log; then the
       launcher puts P to sleep and MEASURES D_c(P) per card.
    5. group D (TP=3 + NEXTN decode, :30032) launches with budgets derived
       from the MEASURED D_c(P); R1 on :30032; D stays awake.
    6. the front (:30030) comes up; deadmen armed for all three logs.

Inherited from /spinning/gpu-arb/boot_855_train0901.sh (named per commit):
the cu13 LD_LIBRARY_PATH line (:152-153, plus S1 killer K1's reason), the
#1217 presence sweep (:107-150), the host-ledger preflight call (:315-319),
the env block (:251-255, :283, :329-330, :333, :358), the argv block
(:689-765 minus the removed flip family), the stale-deadman sweep and the
pgrep proof (:782-821), the mem time series (:829-832) and the
current_boot.log symlink (:865).  The Weg-1 phase-flip family, the seam
flags and the flip image env are NOT inherited (S0 removed the flags).

Declared V1 deviations (all printed at launch and listed in the postmortem):
* transports stay OPEN across sleep -- ``barlink_reopen()`` is not wired on
  the wake path in this round; D's BAR1 windows are sized so both groups fit
  the 3080 aperture (24+96 for P, 16+32+40 for D = 208 of 224 MiB usable;
  #1234 C1 raised dcp:0 from 24, measured BAR1 Used 224/256 per 3080).
* SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 for both groups (K2): /health
  is a pure liveness probe; the front's /health_generate reaches the AWAKE
  group only.
* the deadmen of the two groups run tier 1 only (PROBE_S spaced past the
  boot); tier 2 (/health_generate) runs on the FRONT's deadman, which the
  front routes to the awake group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import host_ledger

MIB = 1024 * 1024
PORT_FRONT = 30030
PORT_P = 30031
PORT_D = 30032
EVIDENCE_DIR = "/spinning/evidence-665-f1"
GPU_ARB = "/spinning/gpu-arb"
DEADMAN = f"{GPU_ARB}/devtools/boot_deadman.sh"
MEMTS = f"{GPU_ARB}/devtools/mem_timeseries.sh"
HOST_PREFLIGHT = f"{GPU_ARB}/devtools/host_ledger_preflight.sh"
PRESENCE_DIR = "/dev/shm/sglang-phase-flip-presence"
STORE_MOUNT = "/spinning/hicache-weg2-ram"
#: The corridor law is 819-1229 MiB NVML-free per card under the awake
#: group's load.  MEASURED 2026-09-07 boot weg2onebackup2 with this constant
#: at 1024: the 5090's continuous minimum under group D was 620-684 MiB
#: (P log "CORRIDOR LAW BREACHED", 4 samples) -- the awake group lands
#: ~400 MiB below the budget line (CUDA context + BAR1 windows sit outside
#: the --rank-gpu-memory-mib fraction).  The measured overshoot is charged
#: here so the minimum lands mid-band; R5 grades the result.
CORRIDOR_MIB = 1024 + 404
#: Spec section 1.6 V1 (arm B, graphs resident) derived upper bounds for the
#: dormant residue of a rank: 5090 1,848 MiB, 3080 1,442 MiB -- EXPECTATIONS
#: (record 1d/1f B6), printed beside the measurement, used only for group P's
#: budget (D has not slept yet when P is sized) and graded by W19 at D's
#: first sleep by the front.  Plus D's open BAR1 windows (deviation above).
DC_EXPECT_5090_MIB = 1848
DC_EXPECT_3080_MIB = 1442
#: MEASURED on boot weg2ls1b2 (2026-09-07 07:11:55-58Z, NVML per-process at
#: group D's first sleep, tags kv_cache+weights, NEXTN draft + TP decode
#: graphs resident): D_c(D) = 2228 MiB on the 5090, 1922 MiB on each 3080 --
#: ABOVE the spec 1.6 expectations by 380 / 480 MiB.  These carry the
#: provenance into P's budget until a boot measures them again; the front's
#: W19 still grades the live measurement against the reserve actually used.
#: SCOPE OF ALL FOUR NUMBERS ABOVE (FIX 1r/1): every boot behind them -- the
#: spec 1.6 expectations and these two measurements -- ran group D with
#: `--disable-overlap-schedule`, hence `no_buffer`, hence ZERO ping-pong mamba
#: state slots. D now runs the overlap schedule, so its device residue carries
#: `d_mamba_ping_pong_cost` extra state slots per rank that no boot behind these
#: constants contained. They are NOT corrected here: a MiB conversion would be a
#: second accounting of the runtime's own sizing (see `d_overlap_cost_line`).
#: The term is printed at boot beside the SCHEDULER line, and W19 grades the
#: live residue rather than these expectations.
DC_MEASURED_D_5090_MIB = 2228
DC_MEASURED_D_3080_MIB = 1922
DC_RESERVE_SLACK_MIB = 64
#: #1234 C6 -- the DEVELOPMENT transport switch, and the one number it moves.
#:
#: The user's order of 2026-09-07 put development on NCCL until barlink
#: covers the Weg-2 message classes again. NCCL is a LAUNCHER-level mode, not
#: a tier inside barlink: a barlink-owned group never constructs a PyNccl
#: communicator (parallel_state.should_build_pynccl), and building one would
#: change the flip path, which is out of bounds. Dropping --barlink instead
#: leaves the stock sglang dispatch in place, which does build one.
#:
#: MEASURED (A/B boot weg2ab0, ARM 0): D's dormant residue on the 5090 rises
#: 2230 -> 2310 MiB under NCCL, because libnccl's buffers are not
#: memory-saver-tagged and therefore survive the sleep. That tripped W19
#: DormantResidueRefused against DC_MEASURED_D_5090_MIB 2228 +
#: DC_RESERVE_SLACK_MIB 64 = 2292 and killed the run. The extra slack lives
#: HERE, behind the switch, and not in the constant -- under 'bar1' nothing
#: about the flip path changes, which is the entire point.
DC_RESERVE_SLACK_NCCL_MIB = 192
#: Flags that only make sense while barlink owns the group's collectives.
#: Each takes a value except the bare --barlink itself.
BARLINK_FLAGS_WITH_VALUE = (
    "--barlink-transport",
    "--barlink-bar1-window-mib",
    "--barlink-bar1-cap-cycles",
    "--barlink-uncovered-class",
)
BARLINK_FLAGS_BARE = ("--barlink",)
#: Environment keys the bar1 build path owns; dropped with the flags so the
#: NCCL arm is not half-configured for a transport it does not run.
BARLINK_ENV_KEYS = ("SGLANG_BARLINK_BUILD_WINDOW_CAP_S",)


def transport_argv(argv: List[str], transport: str) -> List[str]:
    """``argv`` as the chosen transport needs it. One seam, both groups."""
    return argv if str(transport) != "nccl" else strip_barlink_flags(argv)


def reserve_slack_mib(transport: str) -> int:
    """Dormant-residue slack for group D under this transport, in MiB.

    ONE definition, so the NCCL arm's +128 MiB cannot drift away from the
    reason it exists. Under 'bar1' this is the unchanged
    :data:`DC_RESERVE_SLACK_MIB`, and the flip path's arithmetic is exactly
    what it was.
    """
    return (DC_RESERVE_SLACK_NCCL_MIB if str(transport) == "nccl"
            else DC_RESERVE_SLACK_MIB)


def strip_barlink_flags(argv: List[str]) -> List[str]:
    """``argv`` without the barlink family -- the NCCL development mode.

    Removal, not substitution: without --barlink the group takes the stock
    sglang dispatch, which builds the PyNccl communicator barlink suppresses.
    Written as a filter over the ONE argv builder rather than as a second
    argv builder, so the two arms cannot drift apart in anything except the
    transport.
    """
    out: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in BARLINK_FLAGS_WITH_VALUE:
            skip = True
            continue
        if token in BARLINK_FLAGS_BARE:
            continue
        out.append(token)
    return out
#: #1233 boot weg2ls2b2 (2026-09-07 08:58Z, front corridor sampler with group
#: P awake, epoch 1, idle after the wake): NVML free 785 / 1806 / 990 MiB on
#: the 5090 (PP0) / nvml0 (PP1) / nvml2 (PP2) against the budget line's own
#: expectation total - budget - D_c(D) measured = 1501 / 1496 / 1498 MiB, i.e.
#: P lands 716 MiB OVER its line on the 5090 and 508 over on PP2 (PP1 310
#: under).  The 5090 shortfall armed PP0's rank-local #656 narrowing (4096 ->
#: 448) and killed the group (b1/b2 killer).  Charged here PER P ORDINAL so
#: that P's idle free lands ~1.7 GiB on the 5090 (a full 4096-token GDN chunk
#: prices ~635 MiB of transient there; the #656 floor is 1024) and ~1.5 GiB on
#: PP2; PP1 keeps its budget.  R5 grades the result; a boot that measures a
#: different overshoot replaces these numbers, it does not add to them.
P_OVERSHOOT_MIB = [920, 0, 512]
#: #1233 boot weg2ls4b1 (2026-09-07 11:24-11:50Z, front corridor sampler with
#: group D awake under three back-to-back agent-load rounds, 31 requests
#: each): NVML free continuous minimum 535 / 1260 / 1724 MiB on the 5090
#: (TP0, token share 29/64) / nvml0 / nvml2 against the budget line's own
#: expectation of CORRIDOR_MIB = 1428, i.e. D lands 893 MiB OVER its line on
#: the 5090 (NEXTN draft + verify-tree transients and the decode graphs sit
#: outside the --rank-gpu-memory-mib fraction, CAMPAIGN_d0: the capture term
#: is a lower bound ~4x below the driver cost) -- BELOW the 819-1229 band
#: (R5 corridor: UNDER = investigate).  Charged here on ordinal 0 so the
#: 5090's minimum lands mid-band (~1024).  The two 3080 ranks land 168 over /
#: 296 UNDER the line and are NOT charged: their residue is structural (the
#: pool size follows the min-over-ranks token count, so the cheaper ranks
#: allocate less), a planner item (uneven token vector), not a budget one.
#: A boot that measures a different overshoot replaces this number.
D_OVERSHOOT_MIB = [489, 0, 0]
D_WINDOWS_MIB = 16 + 32 + 24
P_WINDOWS_MIB = 24 + 96
MODEL_DEFAULT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
VENV_DEFAULT = "/spinning/htsglang-gpu/.venv"

#: The operating point both groups are launched at (`--context-length`), and
#: therefore the longest prompt group P can be asked to prefill. It is the
#: default floor for P's KV pool: P frees a request's rows once its prefill
#: completes, but DURING that prefill the whole prefix must be device-resident
#: for the stage's attention layers, so a pool below this cannot serve the
#: boot's own admitted maximum.
CONTEXT_LENGTH = 262144

#: MEASURED per-stage prefill cost, ms per layer, boot `bsscale`
#: (/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip 37c884b0b0, table under
#: "The headline of Table A"): per full 4096-token chunk at bs6 PP0 259.1 ms
#: over 32 layers = 8.10, PP1 632.9 over 18 = 35.16, PP2 470.2 over 14 =
#: 33.59, with stage 0 on the 5090 and stages 1-2 on the two 3080s. The two
#: 3080 figures differ by 4.7 %, which is inside the +-10 % per-rank spread
#: that measurement's own A/A repeat established. Used as the ANCHOR for the
#: card-rate library's ratios, and as the whole cost model when no measured
#: library exists.
MEASURED_MS_PER_LAYER = "8.10,35.16,33.59"

#: Per-rank arming floor for the pool model, MiB. The rig's VRAM corridor is
#: 819-1229 MiB NVML-free per card under load and the desk pre-flight arming
#: floor is <= 1229; the ceiling is taken because a floor that under-charges
#: inflates the pool, which is the unsafe direction for a capacity FLOOR.
#: pp_cut.PhasePoolModel names the gap this stands in for: the per-layout
#: solved floor exists only for a layout that has booted.
ARMING_FLOOR_MIB = 1229.0

#: #1240 -- the DEPTH axis of the cost model, and the two records that pin it.
#:
#: A full-attention layer's cost for one chunk grows with the prefix it
#: attends over; a GDN layer's does not. One number per stage cannot carry
#: that, so the family split is solved from TWO measurements at TWO depths
#: (pp_cut.family_costs_from_measurement) and both are named here.
#:
#: CALIBRATION DEPTH. MEASURED_MS_PER_LAYER above was taken on boot bsscale,
#: whose driver sends a ~12,000-token prompt at --chunked-prefill-size 4096,
#: i.e. THREE full chunks (BSSCALE_0907.md: "A ~12 000-token prompt is 3
#: chunks"). Those chunks enter at prefix 0, 4096 and 8192, so the mean prefix
#: of the full chunks the ms/layer figures average over is 4096. Derived, not
#: chosen: it is the arithmetic mean of the census the calibration ran on.
CALIBRATION_PREFIX_TOKENS = 4096

#: DEEP ANCHOR. The user's physics note of 2026-09-07 (recorded verbatim in
#: the GAPPED block of WEG2_BUILD_DECISIONS_0906.md section 1r): at deep
#: prefixes one full-attention layer costs a 3080 about 0.4 s per chunk at a
#: 262,144-token prefix. That is the second depth the family split needs; it
#: is what makes the optimum a FUNCTION of the design prefix rather than a
#: constant, and it is cited to its record rather than written as a literal.
ATTN_ANCHOR_MS = 400.0
ATTN_ANCHOR_PREFIX_TOKENS = 262144

#: DESIGN DEPTH FALLBACK. When no boot log carries a prefill census the design
#: prefix is one chunk -- the shallowest depth the boot can actually run -- and
#: the launcher PRINTS that it fell back, so a table read at 4096 is never
#: mistaken for a table read at this rig's real mean prefix.
DESIGN_PREFIX_FALLBACK_TOKENS = 4096

#: #1240 decode defaults. MEASURED on boot weg2pp1 (arms table row D2 of
#: /spinning/gpu-arb/weg2/BOOT_weg2pp1_0907.md): overlap ON with
#: --num-continuous-decode-steps 2 and the paired extra_buffer mamba strategy
#: reads 75.5 tok/s at bs1 and 305.0 at bs6, against the D0 control's 66.2 /
#: 284.9 -- +14.0 % / +7.1 %. Row D3 (steps 4) REGRESSES to 73.6 / 295.6, so
#: 2 is an optimum and not a direction. The steps knob is the unconfounded
#: half of that boot (D1 -> D2 -> D3 all run overlap ON and extra_buffer).
D_NUM_CONTINUOUS_DECODE_STEPS = 2


class Weg2LaunchRefused(RuntimeError):
    pass


@dataclass
class Card:
    nvml_index: int
    uuid: str
    name: str
    total_mib: int


@dataclass
class GroupSpec:
    name: str
    port: int
    argv: List[str]
    log: str
    env: Dict[str, str]
    pid: int = 0
    proc: Optional[subprocess.Popen] = None


@dataclass
class BootState:
    tag: str
    tip: str
    tree: str
    stamp: str
    cards: List[Dict] = field(default_factory=list)
    cvd: str = ""
    logs: Dict[str, str] = field(default_factory=dict)
    pids: Dict[str, int] = field(default_factory=dict)
    helper_pids: List[int] = field(default_factory=list)
    store_mount: str = STORE_MOUNT
    store_gib: float = 0.0
    budgets: Dict[str, List[int]] = field(default_factory=dict)
    dc_measured_p: Dict[str, int] = field(default_factory=dict)
    dc_expect_d: Dict[str, int] = field(default_factory=dict)
    ledger_lines: List[str] = field(default_factory=list)
    argv: Dict[str, str] = field(default_factory=dict)
    t_ready: Dict[str, float] = field(default_factory=dict)
    sleep_p_ms: float = 0.0
    deviations: List[str] = field(default_factory=list)
    carrier_max_tokens: int = 0
    weight_chunks: int = 0
    tms_so: str = ""
    p_depth: int = 0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{_now()}] WEG2-LAUNCH {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()


# --------------------------------------------------------------------------
# NVML
# --------------------------------------------------------------------------


def resolve_cards() -> List[Card]:
    import pynvml

    pynvml.nvmlInit()
    try:
        n = pynvml.nvmlDeviceGetCount()
        cards = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            cards.append(Card(i, uuid, name, int(mem.total // MIB)))
        return cards
    finally:
        pynvml.nvmlShutdown()


def order_cards(cards: List[Card]) -> List[Card]:
    """CUDA ordinal order: the 5090 first (rank 0 / PP0 / TP0), then the 3080s
    by NVML index.  Never a fixed index: resolved by name per launch."""
    big = [c for c in cards if "5090" in c.name]
    small = sorted([c for c in cards if "3080" in c.name], key=lambda c: c.nvml_index)
    if len(big) != 1 or len(small) != 2:
        raise Weg2LaunchRefused(
            f"NVML inventory is not 1x5090 + 2x3080: {[(c.nvml_index, c.name) for c in cards]}"
        )
    return [big[0], small[0], small[1]]


def nvml_used_free(cards: List[Card]) -> Dict[str, Tuple[int, int]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    res: Dict[str, Tuple[int, int]] = {}
    for line in out.strip().splitlines():
        u, used, total = [x.strip() for x in line.split(",")]
        res[u] = (int(used), int(total))
    return res


def nvml_process_mib(pids: set) -> Dict[str, int]:
    """Sum of NVML per-process usage per card UUID over ``pids``."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    res: Dict[str, int] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, used, uuid = [x.strip() for x in line.split(",")]
        if int(pid) in pids:
            res[uuid] = res.get(uuid, 0) + int(used)
    return res


def session_pids(sid: int) -> set:
    out = subprocess.run(["ps", "-eo", "pid,sid"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == sid:
            pids.add(int(parts[0]))
    return pids


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


def http(method: str, url: str, body: Optional[dict] = None, timeout: float = 25.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def wait_ready(port: int, pid: int, deadline_s: float, log: Log, name: str, proc: Optional[subprocess.Popen] = None) -> float:
    t0 = time.time()
    last = ""
    while time.time() - t0 < deadline_s:
        # proc.poll() reaps: a dead child is a ZOMBIE until reaped and
        # os.kill(pid, 0) still succeeds on it (measured 07:00Z: D died at
        # parse and the launcher kept waiting).
        if proc is not None and proc.poll() is not None:
            raise Weg2LaunchRefused(f"group {name} pid {pid} died before READY (exit {proc.returncode}); see its log")
        if pid and not _alive(pid):
            raise Weg2LaunchRefused(f"group {name} pid {pid} died before READY")
        code, body = http("GET", f"http://127.0.0.1:{port}/health", timeout=25)
        if code == 200:
            dt = time.time() - t0
            log(f"R1 READY group={name} port={port} after {dt:.1f} s")
            return dt
        last = f"{code} {body[:80]!r}"
        time.sleep(3)
    raise Weg2LaunchRefused(f"group {name} not READY on :{port} after {deadline_s:.0f} s (last {last})")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Preflight (inherited from boot_855_train0901.sh)
# --------------------------------------------------------------------------


def presence_sweep(log: Log, tag: str, stamp: str, dry: bool) -> None:
    if not os.path.isdir(PRESENCE_DIR):
        log("#1217 presence residue: none")
        return
    files = os.listdir(PRESENCE_DIR)
    if not files:
        log("#1217 presence residue: none")
        return
    live = subprocess.run(["pgrep", "-f", "sglang[.]launch_server"], capture_output=True, text=True).stdout.split()
    held = subprocess.run(["fuser", "-s"] + [os.path.join(PRESENCE_DIR, f) for f in files], capture_output=True).returncode == 0
    if live or held:
        raise Weg2LaunchRefused(
            f"#1217 presence residue: LIVE HOLDER -- refusing this boot, not sweeping, not killing anything. "
            f"launch_server pid(s)=[{' '.join(live) or 'none'}] fuser_held={held}"
        )
    archive = f"{GPU_ARB}/shm_residue/{tag}_{stamp}"
    if dry:
        log(f"#1217 DRY-RUN: would sweep {len(files)} file(s) from {PRESENCE_DIR} -> {archive}")
        return
    os.makedirs(archive, exist_ok=True)
    for f in files:
        shutil.move(os.path.join(PRESENCE_DIR, f), archive)
    log(f"#1217 presence residue swept: {len(files)} file(s) from {PRESENCE_DIR} -> {archive}")


def stale_deadman_sweep(log: Log, ports: Sequence[int], dry: bool) -> None:
    out = subprocess.run(["pgrep", "-af", "boot_deadman.sh"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, argv = line.partition(" ")
        if any(f" {p}" in argv for p in ports) and "boot_deadman.sh" in argv:
            if dry:
                log(f"DRY-RUN: would kill stale deadman pid {pid} ({argv})")
            else:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    log(f"stale deadman pid {pid} killed (argv: {argv}) -- it watched a previous boot")
                except OSError:
                    pass


def host_preflight(log: Log, tag: str, dry: bool) -> None:
    if dry:
        log("DRY-RUN: host_ledger_preflight.sh not run")
        return
    rc = subprocess.run([HOST_PREFLIGHT, f"{GPU_ARB}/preflight_weg2_{tag}.log"]).returncode
    if rc != 0:
        raise Weg2LaunchRefused("BOOT REFUSED by the host-ledger preflight (#721 floor) -- see its log")
    mi = host_ledger.read_meminfo()
    avail_gib = mi["MemAvailable"] / host_ledger.GIB
    if avail_gib < 40:
        top = subprocess.run(["ps", "-eo", "rss,pid,comm", "--sort=-rss"], capture_output=True, text=True).stdout.splitlines()[:8]
        raise Weg2LaunchRefused(f"free -g available {avail_gib:.1f} GiB < 40 GiB; top RSS holders (NOT killed):\n" + "\n".join(top))
    oom = open("/sys/fs/cgroup/memory.events").read()
    log(f"host preflight PASS: MemAvailable {avail_gib:.1f} GiB; cgroup memory.events baseline: {' '.join(oom.split())}")


def cards_free_check(cards: List[Card], log: Log) -> None:
    uf = nvml_used_free(cards)
    for c in cards:
        used, total = uf[c.uuid]
        if used > 1500:
            raise Weg2LaunchRefused(f"card {c.nvml_index} ({c.name}) has {used} MiB used (> 1500) -- not free; not killing anything")
    log("cards free: " + ", ".join(f"idx{c.nvml_index}={uf[c.uuid][0]}/{uf[c.uuid][1]} MiB used" for c in cards))


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


def mount_store(log: Log, store_gib: float, dry: bool) -> None:
    if dry:
        log(f"DRY-RUN: would mount tmpfs size={store_gib:.0f}G at {STORE_MOUNT}")
        return
    os.makedirs(STORE_MOUNT, exist_ok=True)
    mounts = open("/proc/mounts").read()
    if f" {STORE_MOUNT} " in mounts:
        subprocess.run(["umount", STORE_MOUNT], check=False)
        log(f"store: stale tmpfs at {STORE_MOUNT} unmounted (fresh store per boot)")
    subprocess.run(["mount", "-t", "tmpfs", "-o", f"size={int(store_gib)}G", "tmpfs", STORE_MOUNT], check=True)
    st = os.statvfs(STORE_MOUNT)
    log(
        f"store: tmpfs mounted at {STORE_MOUNT} size={st.f_blocks * st.f_frsize / host_ledger.GIB:.2f} GiB "
        f"(RAM-backed canonical page store, user ruling 2026-09-07; NO disk tier in V1 -- the file "
        f"backend has no second tier, /spinning/hicache-weg2 is not used)"
    )


def store_extra_config(store_gib: float) -> str:
    # max_size + min_free_space must fit the filesystem (W8): the tmpfs IS
    # store_gib, so max_size = store_gib - 1 G and min_free_space = 1 G.
    max_size = max(1, int(store_gib) - 1)
    return json.dumps({"max_size": f"{max_size}G", "min_free_space": "1G", "max_size_scope": "shared"}, separators=(",", ":"))


# --------------------------------------------------------------------------
# argv composition
# --------------------------------------------------------------------------


def common_flags(
    model: str,
    s_gb: int,
    m_mib: int,
    store_gib: float,
    write_policy: str = "write_through",
) -> List[str]:
    # --disable-overlap-schedule IS NO LONGER HERE. It is group P's flag, not
    # a common one: the justification is `pp_size > 1` (server_args.py:19507,
    # "Pipeline parallelism is not compatible with overlap schedule", plus
    # the same forcing in arg_groups/overrides._pipeline_parallel_overlap_
    # disable), and group D runs pp_size=1. MEASURED consequence of the old
    # placement: D booted with disable_overlap_schedule=True inheriting a
    # PP-only reason (BSSCALE_0907.md D4), so CPU scheduling of round n+1
    # could not hide behind GPU work of round n on a group that has no
    # pipeline at all. See argv_p / argv_d.
    return [
        "--model-path", model,
        "--trust-remote-code",
        "--served-model-name", "Qwen3.8-27B",
        "--rank-gpu-id", "0,1,2",
        "--skip-server-warmup",
        "--kv-cache-dtype", "fp8_e4m3",
        "--context-length", str(CONTEXT_LENGTH),
        "--max-running-requests", "8",
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_coder",
        "--chat-template-default-kwargs", '{"preserve_thinking": true}',
        "--enable-cache-report",
        "--enable-metrics",
        "--enable-hierarchical-cache",
        "--hicache-host-role", "staging",
        "--hicache-size", str(s_gb),
        "--hicache-mamba-host-mib", str(m_mib),
        "--hicache-write-policy", write_policy,
        "--hicache-storage-backend", "file",
        "--hicache-mem-layout", "layer_first",
        "--hicache-io-backend", "direct",
        "--hicache-storage-backend-extra-config", store_extra_config(store_gib),
        "--hicache-canonical-kv-page",
        "--host", "127.0.0.1",
        "--chunked-prefill-size", "4096",
        "--scheduler-distributed-teardown",
        "--page-size", "1",
        "--random-seed", "785500001",
        "--mamba-ssm-dtype", "bfloat16",
        "--mamba-slot-reorder",
        "--kv-backing-relief",
        "--barlink", "--barlink-transport", "bar1",
        "--barlink-bar1-cap-cycles", "300000000000",
        "--collective-census-interval", "50",
        "--enable-memory-saver",
        "--enable-weights-cpu-backup",
    ]


def argv_p(
    py: str,
    model: str,
    budgets: List[int],
    s_gb: int,
    m_mib: int,
    store_gib: float,
    extra: List[str],
    stage_ratio: str = "32,18,14",
    attn_stage_ratio: str = "8,4,4",
    write_policy: str = "write_through",
    depth: int = 0,
) -> List[str]:
    # THE COUNT FLAGS ARE THE CONTIGUOUS FORM, AND ONLY THAT (#1240 FOLLOW FIX
    # 1). --pp-stage-ratio/--pp-attn-stage-ratio are per-stage COUNTS that
    # server_args hands to derive_pp_layer_split, which builds a CONTIGUOUS
    # split from them. A gapped map is not expressible that way at all:
    # * the user's own layout (48 GDN on the 5090, 8+8 attention on the 3080s)
    #   is `48,8,8 / 0,8,8`, and derive_pp_layer_split raises
    #   "--pp-attn-stage-ratio entries must be positive integers" before a
    #   single weight loads -- boot weg2pp2 arm P_G1 died exactly there;
    # * a gapped map whose attention counts are all positive is worse than
    #   refused, it is ACCEPTED and derives something else (52,6,6 / 4,6,6 ->
    #   [19,24,21]), so the argv would contradict SGLANG_PP_LAYER_SET.
    # Under a gapped map the SET published in env_p IS the layout and
    # make_layers/model_runner resolve ownership from get_pp_layer_set, never
    # from the count form -- so the count copy is the second set of books and
    # it is dropped, rather than kept and asserted against.
    if bool(stage_ratio) != bool(attn_stage_ratio):
        raise Weg2LaunchRefused(
            "W40 Weg2PPCutRefused: --pp-stage-ratio %r and "
            "--pp-attn-stage-ratio %r must be given together or omitted "
            "together. They are one statement of one layout; half of it "
            "would let derive_pp_layer_split snap the other half silently "
            "(the #505(a) class)." % (stage_ratio, attn_stage_ratio)
        )
    # SOLVED unless the operator pinned them; the launcher prints the PP-CUT
    # provenance line either way. Never a hand constant reaching this line
    # unannounced -- and, since FOLLOW FIX 1, never a SECOND statement of a
    # layout the --pp-layer-set wire already carries.
    ratio_flags = (
        ["--pp-stage-ratio", stage_ratio, "--pp-attn-stage-ratio", attn_stage_ratio]
        if stage_ratio
        else []
    )
    return [py, "-m", "sglang.launch_server"] + common_flags(model, s_gb, m_mib, store_gib, write_policy) + [
        "--tp-size", "1", "--pp-size", "3",
        # #692 MICROBATCH DEPTH, group P only -- group D runs pp_size=1 and a
        # pipeline depth is meaningless there. STATED even at 0 so the argv is
        # an honest statement of what the boot runs, and published ONCE as a
        # group constant: every rank reads the same token off the same argv,
        # so no rank can derive a different depth (the ring size is a
        # collective property -- two ranks disagreeing about pp_loop_size is
        # the v7pp12 starvation, not a tuning difference). Solved by
        # solve_p_depth from the previous boot's own PP-BUBBLE line; the
        # launcher prints the WEG2 P-DEPTH provenance line either way.
        "--pp-async-batch-depth", str(int(depth)),
        # P is the pp_size>1 group, so the overlap schedule is refused HERE
        # and only here (server_args.py:19507). Passing it explicitly rather
        # than letting the post-process pass force it keeps the argv an
        # honest statement of what this group runs.
        "--disable-overlap-schedule",
    ] + ratio_flags + [
        "--rank-gpu-memory-mib", ",".join(str(b) for b in budgets),
        "--barlink-bar1-window-mib", "24,PP_0=96",
        "--port", str(PORT_P),
    ] + extra


def argv_d(
    py: str,
    model: str,
    budgets: List[int],
    s_gb: int,
    m_mib: int,
    store_gib: float,
    extra: List[str],
    num_continuous_decode_steps: int = 1,
    disable_overlap: bool = False,
) -> List[str]:
    return [py, "-m", "sglang.launch_server"] + common_flags(model, s_gb, m_mib, store_gib) + (
        ["--disable-overlap-schedule"] if disable_overlap else []
    ) + [
        # THE STRATEGY IS STATED, NOT INHERITED (FIX 1r/1). Leaving this at
        # 'auto' made D's mamba radix-cache strategy a SIDE EFFECT of the
        # absent disable flag: _mamba_radix_cache_resolution reads
        # `wants_overlap = not view.disable_overlap_schedule`
        # (arg_groups/overrides.py:1225-1239), so switching the overlap
        # schedule on also switched no_buffer -> extra_buffer, and
        # mamba_pool_floor.mamba_ping_pong_slots then charges 2 state slots
        # per running request instead of 0 -- DOUBLING D's device mamba floor
        # (16 -> 32 slots at --max-running-requests 8) out of the FIXED
        # --rank-gpu-memory-mib budgets. Stated here so the argv is an honest
        # statement of what D runs, and priced in the SCHEDULER: log line
        # (d_overlap_cost_line) so the launcher cannot move D's device
        # residency without the number appearing. The value follows the
        # overlap choice exactly -- it is not a second knob.
        "--mamba-radix-cache-strategy", "no_buffer" if disable_overlap else "extra_buffer",
        "--tp-size", "3", "--pp-size", "1",
        # OVERLAP SCHEDULE ON for D -- by ABSENCE of the disable flag, which
        # is the only way to have it: there is no --enable-overlap-schedule.
        # Every gate that forces it off was checked against THIS argv and
        # none applies: pp_size>1 (D is 1), --enable-pdmux (not passed,
        # server_args.py:19553), device cpu/mps (cuda), sparse-head
        # embeddings and dllm (neither), and the hybrid-mamba resolution
        # (arg_groups/overrides._mamba_radix_cache_resolution), which picks
        # 'extra_buffer' and LEAVES overlap on for Qwen3_5ForConditional-
        # Generation on linear_attn_backend=triton -- that arch is in
        # _MAMBA_EXTRA_BUFFER_ARCHS. If a gate ever does refuse, it is not to
        # be weakened: the launcher's --d-disable-overlap-schedule puts the
        # flag back and logs W41.
        #
        # #1030's justification is PP-only and does not reach D
        # (BSSCALE_0907.md D4).
        "--num-continuous-decode-steps", str(int(num_continuous_decode_steps)),
        "--rank-gpu-memory-mib", ",".join(str(b) for b in budgets),
        # a per-rank MiB LIST under TP requires the uneven-TP ratio; 'auto'
        # derives the weights from that list (server_args.py rank_tp_ratio).
        "--rank-tp-ratio", "auto",
        "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "2",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "3",
        "--uneven-dcp", "--uneven-dcp-weighted",
        "--uneven-token-vector", "29,19,16", "--uneven-token-vector-role", "seed",
        # #1234 C1 -- dcp:0 goes 24 -> 40 MiB. MEASURED, and this one
        # number is the whole regression fix.
        #
        # WHY 24 WAS WRONG. At a 24-MiB window max_payload yields
        # chunk_max = 2 093 056 B, exactly one 4096-byte page below 2 MiB,
        # so 16 rounds carry 100 466 688 B = 4088 tokens. The dcp
        # attention-out combine at --chunked-prefill-size 4096 sends
        # 4096 x 24576 = 100 663 296 B. Miss: 196 608 B, 0.196 %, EIGHT
        # TOKENS -- and every one of the 16 calls per prefill chunk ran on
        # the host-staged gloo plane at 0.68 GB/s instead of bar1's 3.19.
        #
        # WHY 40 AND NOT MORE. The binding gate is not "sum of declared <=
        # 224" but NVML free minus RESERVE_MIB_DEFAULT (32), evaluated when
        # each group builds (barlink_matrix_transport.py:330, :357-364);
        # measured at dcp:0 build time on boot weg2ab1: 74 - 32 = 42 MiB.
        # DCP_0=48 was REFUSED on metal (Bar1WindowRefused, boot weg2ab1).
        # At 40 MiB chunk_max = 3 493 888, so the 96-MiB all_reduce plans to
        # 10 rounds and the 22-MiB q_full all_gather to 7; measured BAR1
        # Used 224/256 MiB per 3080, i.e. the aperture is EXHAUSTED, which
        # is the argument for the derived round bound rather than for a
        # bigger window next time.
        #
        # WHY NOT 25 (the arithmetic minimum for 16 rounds). Because the
        # window also buys all_gather rounds: 12 -> 7 across the same step.
        # ARM 1 of the A/B boot measured w40 at 2361.4 ms of wait per full
        # chunk against the baseline's 4755.7 and NCCL's 3307.6.
        "--barlink-bar1-window-mib", "16,TP_0=32,DCP_0=40",
        # #1234 C5: group D is the one place where a silent host-staged
        # 4.7x is a known boot killer, and with the window above no declared
        # class is anywhere near the round budget. The library default stays
        # 'warn'; opting in is a deployment decision, made here.
        #
        # WHAT THIS DOES AND DOES NOT STOP (FIX 1). Only the two size-driven
        # refusal kinds -- 'round' and 'oversize', barlink.py's
        # UNCOVERED_REFUSAL_STOPS -- stop the group. A sub-min_bytes or
        # misaligned collective anywhere in D's process still takes the
        # priced warn path and its small, correct gloo answer; killing a boot
        # for one of those would be an abort wider than the reason above.
        # Under the DERIVED bound the 'round' half is currently inert on this
        # rig (the crossover is scale-invariant and has no root here -- see
        # barlink_bar1.round_budget); it binds when a cap is pinned, which is
        # exactly how boot weg2zr2 failed, and 'oversize' is the reachable
        # half that guards the window this line sits next to.
        "--barlink-uncovered-class", "refuse",
        "--port", str(PORT_D),
    ] + extra


def build_env(tree: str, venv: str, cvd: str, store_dir: str, debug_hold: bool, tag: str,
              chunk_layers: int = 0, chunk_count: int = 0, tms_so: str = "",
              transport: str = "bar1") -> Dict[str, str]:
    env = dict(os.environ)
    # #1233 one-backup flip: chunked weights tags (weg2_memory_saver.py) and
    # the patched torch_memory_saver preload hook (tms_csrc/PATCH.md).
    if chunk_layers > 0 and chunk_count > 0:
        env["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = str(chunk_layers)
        env["SGLANG_WEG2_WEIGHT_CHUNKS"] = str(chunk_count)
    if tms_so:
        env["SGLANG_WEG2_TMS_PRELOAD_SO"] = tms_so
    cu13 = f"{venv}/lib/python3.12/site-packages/nvidia/cu13/lib"
    # boot_855_train0901.sh:152-153 (NVRTC) and S1 killer K1: the memory
    # saver's cu13 preload hook links libcudart.so.13, which must be on the
    # loader path at LD_PRELOAD time or every rank dies exit 127 before main().
    env["LD_LIBRARY_PATH"] = cu13 + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = cvd
    env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"  # K2: pure liveness /health
    # #1233 boot weg2ls3b3: group D (NEXTN) keys the store by BIGRAM page
    # hashes, group P (no spec) by UNIGRAM -- disjoint chains for the same
    # prompt, D never read P's pages. One key scheme for both groups; a
    # no-op on D (already bigram), forces bigram on P.
    env["SGLANG_HICACHE_BIGRAM_KEYS"] = "1"
    # #1233 zero-remainder: /flush_cache (the front's quiesce before a sleep)
    # first publishes every un-backed device node to the store, so a chain
    # the write-through pin budget declined mid-prefill is not lost at the
    # flip (UnifiedRadixCache.publish_unbacked_sweep). Both groups.
    env["SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP"] = "1"
    env["SGLANG_ARMING_FLOOR_SOLVED"] = "1"
    env["SGLANG_UNEVEN_DCP"] = "1"
    env["SGLANG_UNEVEN_DCP_WEIGHTED"] = "1"
    env["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
    env["SGLANG_BARLINK_BUILD_WINDOW_CAP_S"] = env.get("SGLANG_BARLINK_BUILD_WINDOW_CAP_S", "60")
    if str(transport) == "nccl":
        # #1234 C6: half-configuring a transport the group does not run is
        # how a mode switch turns into a mystery. The flags go with
        # strip_barlink_flags(), the env keys go here.
        for key in BARLINK_ENV_KEYS:
            env.pop(key, None)
    env["SGLANG_PP_CHAIN_RECV_STALL_S"] = env.get("SGLANG_PP_CHAIN_RECV_STALL_S", "60")
    env["SGLANG_PP_OCCUPANT_HORIZON_S"] = env.get("SGLANG_PP_OCCUPANT_HORIZON_S", "90")
    env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = store_dir
    env["SGLANG_MATCH_REFUSAL_CENSUS_EVERY"] = env.get("SGLANG_MATCH_REFUSAL_CENSUS_EVERY", "64")
    for k in list(env):
        if k.startswith("SGLANG_PHASE_FLIP"):
            del env[k]
    alloc = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc:
        raise Weg2LaunchRefused(
            f"PYTORCH_CUDA_ALLOC_CONF={alloc!r} -- torch_memory_saver refuses expandable_segments (entrypoint.py _sanity_checks)"
        )
    env.pop("SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION", None)
    if debug_hold:
        env["SGLANG_DEBUG_HOLD"] = "1"
        env["SGLANG_DEBUG_HOLD_S"] = env.get("SGLANG_DEBUG_HOLD_S", "1800")
        env["SGLANG_DEBUG_HOLD_PORT_BASE"] = env.get("SGLANG_DEBUG_HOLD_PORT_BASE", "5000")
        env["SGLANG_DEBUG_HOLD_DIR"] = env.get("SGLANG_DEBUG_HOLD_DIR", f"{GPU_ARB}/debug_hold")
        env["SGLANG_DEBUG_HOLD_TAG"] = tag
        os.makedirs(env["SGLANG_DEBUG_HOLD_DIR"], exist_ok=True)
    return env


def count_marker(path: str, marker: str) -> int:
    n = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if marker in line:
                    n += 1
    except FileNotFoundError:
        pass
    return n


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------


def launch_group(spec: GroupSpec, tree: str, log: Log, dry: bool) -> None:
    log(f"group {spec.name} argv: " + " ".join(shlex.quote(a) for a in spec.argv))
    if dry:
        return
    fh = open(spec.log, "ab")
    fh.write(f"=== WEG2 group {spec.name} launched {_now()} ===\nargv: {' '.join(shlex.quote(a) for a in spec.argv)}\n".encode())
    fh.flush()
    p = subprocess.Popen(spec.argv, env=spec.env, stdout=fh, stderr=subprocess.STDOUT, cwd=tree, start_new_session=True)
    spec.pid = p.pid
    spec.proc = p
    log(f"group {spec.name} pid {p.pid} (session id = pid) log {spec.log}")


def arm_deadman(log: Log, boot_log: str, port: int, pattern: str, probe_s: int, tag: str, name: str, dry: bool) -> int:
    out = f"{GPU_ARB}/deadman_{tag}_{name}.out"
    cmd = f"GRACE_S=600 PROBE_S={probe_s} setsid {DEADMAN} {shlex.quote(boot_log)} {port} {shlex.quote(pattern)} > {shlex.quote(out)} 2>&1 & echo $!"
    if dry:
        log(f"DRY-RUN: would arm deadman: {cmd}")
        return 0
    pid = int(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip() or 0)
    time.sleep(1)
    proof = subprocess.run(["pgrep", "-af", f"boot_deadman.sh {boot_log}"], capture_output=True, text=True).stdout.strip()
    n = len(proof.splitlines())
    log(f"deadman {name}: pid {pid} GRACE_S=600 PROBE_S={probe_s} pattern={pattern!r} verdict -> {out}; pgrep proof: {n} process(es) whose argv carries THIS log ({proof or 'NONE -- UNKNOWN, never alive'})")
    return pid


def sleep_group(port: int, log: Log, name: str, weights_tags: List[str]) -> float:
    tags = ["kv_cache"] + list(weights_tags)
    t0 = time.time()
    code, body = http("POST", f"http://127.0.0.1:{port}/release_memory_occupation", {"tags": tags}, timeout=900)
    dt = (time.time() - t0) * 1000
    if code != 200:
        raise Weg2LaunchRefused(f"sleep({name}) failed: HTTP {code} {body[:300]!r}")
    log(f"sleep({name}) OK in {dt:.0f} ms (tags {','.join(tags)}; flush BEFORE pause per record 1d MUST_FIX)")
    return dt


def model_num_layers(model: str) -> int:
    with open(os.path.join(model, "config.json")) as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    n = int(text.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    if n <= 0:
        raise Weg2LaunchRefused(f"num_hidden_layers not found in {model}/config.json")
    return n


def build_tms_preload(tree: str, venv: str, log: Log) -> str:
    """Build (or reuse) the patched torch_memory_saver preload hook."""
    script = os.path.join(tree, "scripts", "weg2", "tms", "build_tms_preload.sh")
    r = subprocess.run([script, "--venv", venv], capture_output=True, text=True)
    if r.returncode != 0:
        raise Weg2LaunchRefused(f"torch_memory_saver preload build failed: {r.stderr[-800:]}")
    so = r.stdout.strip().splitlines()[-1]
    log(f"torch_memory_saver 0.0.9.post1 preload hook REBUILT from python/sglang/srt/weg2/tms_csrc (PATCH.md: cpu backup "
        f"freed after resume) -> {so}; SGLANG_WEG2_TMS_PRELOAD_SO set for both groups")
    return so


def budgets_from_dc(
    cards: List[Card],
    dc_mib: Dict[str, int],
    log: Log,
    label: str,
    overshoot_mib: Optional[List[int]] = None,
    overshoot_provenance: str = "",
) -> List[int]:
    out = []
    for i, c in enumerate(cards):
        over = int(overshoot_mib[i]) if overshoot_mib is not None else 0
        b = c.total_mib - CORRIDOR_MIB - dc_mib[c.uuid] - over
        b = (b // 8) * 8
        out.append(b)
        log(
            f"budget {label} ordinal={i} nvml_idx={c.nvml_index} {c.name}: "
            f"{b} MiB = total {c.total_mib} - corridor {CORRIDOR_MIB} - dormant_other {dc_mib[c.uuid]}"
            + (f" - measured_awake_overshoot {over} ({overshoot_provenance})" if over else "")
            + " MiB"
        )
    return out


def _max_running_requests(model: str) -> int:
    """``--max-running-requests`` as this launcher actually passes it."""
    flags = common_flags(model, 1, 1, 1.0)
    return int(flags[flags.index("--max-running-requests") + 1])


def d_mamba_ping_pong_cost(model: str, disable_overlap: bool) -> Tuple[str, int, int, int]:
    """What D's overlap choice costs in DEVICE mamba state slots (FIX 1r/1).

    Returns ``(strategy, ping_pong_slots_per_running_request, extra_slots_per
    _rank, max_running_requests)``.

    THE TERM. ``arg_groups/overrides._mamba_radix_cache_resolution`` reads
    ``wants_overlap = not view.disable_overlap_schedule``, so turning the
    overlap schedule on for group D ALSO turns its mamba radix-cache strategy
    from ``no_buffer`` to ``extra_buffer`` -- and
    ``mem_cache/mamba_pool_floor.mamba_ping_pong_slots`` then charges 2 slots
    per running request where it charged 0. That is device residency out of
    D's FIXED ``--rank-gpu-memory-mib`` budgets on a rig whose corridor law is
    819-1229 MiB NVML-free per card, and the commit that switched the overlap
    schedule on priced only the overlap benefit. An unpriced term does not
    read as unknown, it reads as free (#1009).

    NOT A SECOND ACCOUNTING. The slot count is taken from the runtime's own
    ``mamba_ping_pong_slots`` against a view carrying exactly the three fields
    it reads, with ``ServerArgs.enable_mamba_extra_buffer`` itself as the
    predicate -- so a change to either function moves this number too. Only
    the PING-PONG term is charged, because it is the only term of
    ``mamba_slots_per_running_req`` that depends on the overlap choice; the
    active slot and the donation/pin term are identical in both arms and
    cancel in the delta. The remaining terms are deliberately NOT reproduced
    here: ``mamba_slot_reorder_active`` reads an environment variable, and the
    launcher's environment is not the group's (``build_env``), so evaluating
    it here would answer about the wrong process.
    """
    from sglang.srt.mem_cache.mamba_pool_floor import mamba_ping_pong_slots
    from sglang.srt.server_args import ServerArgs

    flags = common_flags(model, 1, 1, 1.0)

    class _StrategyView:
        """Exactly the ServerArgs surface ``mamba_ping_pong_slots`` reads."""

        # The upstream predicate itself, not a restatement of it.
        enable_mamba_extra_buffer = ServerArgs.enable_mamba_extra_buffer

        def __init__(self, strategy: str, disable_overlap_schedule: bool) -> None:
            self.mamba_radix_cache_strategy = strategy
            self.disable_overlap_schedule = disable_overlap_schedule
            # Read off the argv this launcher builds, never asserted: the day
            # --disable-radix-cache appears in common_flags the price changes
            # to 0 and this follows it.
            self.disable_radix_cache = "--disable-radix-cache" in flags

    strategy = "no_buffer" if disable_overlap else "extra_buffer"
    per_req = mamba_ping_pong_slots(_StrategyView(strategy, disable_overlap))
    # The arm this replaces: group D as it booted before the overlap schedule
    # was turned on, i.e. the arm every DC_MEASURED_D_* number was taken on.
    baseline = mamba_ping_pong_slots(_StrategyView("no_buffer", True))
    mrr = _max_running_requests(model)
    return strategy, per_req, (per_req - baseline) * mrr, mrr


def d_overlap_cost_line(model: str, disable_overlap: bool) -> str:
    """The one line that must appear wherever D's overlap choice is announced.

    The launcher must not be able to change D's device residency without the
    number appearing (FIX 1r/1), so the price is built from
    :func:`d_mamba_ping_pong_cost` rather than typed, and the MiB conversion
    it does NOT make is named rather than left as a silent omission.
    """
    strategy, per_req, extra, mrr = d_mamba_ping_pong_cost(model, disable_overlap)
    baseline_per_req = per_req - (extra // max(1, mrr))
    return (
        f"DEVICE PRICE OF THAT CHOICE: --mamba-radix-cache-strategy "
        f"{strategy} is now STATED on D's argv instead of falling out of the "
        f"absent disable flag (arg_groups/overrides._mamba_radix_cache_"
        f"resolution reads `wants_overlap = not view.disable_overlap_"
        f"schedule`). Derived from the runtime's own mamba_pool_floor."
        f"mamba_ping_pong_slots against the no_buffer arm this replaces: "
        f"{per_req} - {baseline_per_req} = {per_req - baseline_per_req} extra "
        f"ping-pong state slots per running request x --max-running-requests "
        f"{mrr} = {extra} extra device mamba state slots on EVERY D rank, out "
        f"of the FIXED --rank-gpu-memory-mib budgets and inside the "
        f"819-1229 MiB corridor. This term is not converted to MiB here and "
        f"the refusal is named: per-rank slot bytes follow Mamba2StateShape "
        f"under --rank-tp-ratio auto, and re-deriving that shape in the "
        f"launcher would be a second accounting of the runtime's own sizing; "
        f"the boot prints it as 'mamba_cache_per_req=<x> MB' "
        f"(model_runner_kv_cache_mixin.py:2529) and the front's W19 grades "
        f"the live residue. DC_EXPECT_*/DC_MEASURED_D_* were measured on "
        f"no_buffer boots and do NOT contain this term."
    )


#: The one line ``read_pp_bubble`` understands, emitted per window per rank by
#: ``scheduler_components/pp_bubble.py:summary_line``. Anchored on the whole
#: field name including its ``=``: a bare number would match milliseconds
#: elsewhere in the line, which is the bare-ticket-number trap in miniature.
_BUBBLE_RE = re.compile(
    r"PP-BUBBLE rank=(?P<rank>\d+) .*?"
    r" n=(?P<n>\d+) \(n_gaps=(?P<n_gaps>\d+),"
    r" forward_ms=(?P<forward>[0-9.]+),"
    r" bubble_ms=(?P<bubble>[0-9.]+),"
    r" starved_ms=(?P<starved>[0-9.]+);"
)


@dataclass(frozen=True)
class BubbleMeasurement:
    """One stage's PP-BUBBLE totals, summed over every window in one log.

    Summed, never averaged: each window line carries its own numerator AND its
    own denominator, so a mean of the printed shares would weight a 12-forward
    window like a 2-forward one. Sum of numerators over sum of denominators is
    the only aggregation that keeps the denominator honest.
    """

    source: str
    rank: int
    windows: int
    forward_ms: float
    bubble_ms: float
    starved_ms: float
    n_forwards: int

    @property
    def denominator_ms(self) -> float:
        """``gap + forward`` -- the denominator the emitting line names."""
        return self.bubble_ms + self.forward_ms

    @property
    def bubble_share(self) -> float:
        d = self.denominator_ms
        return 0.0 if d <= 0.0 else self.bubble_ms / d

    @property
    def forward_share(self) -> float:
        return 1.0 - self.bubble_share

    @property
    def stall_share(self) -> float:
        """The part depth can move.

        ``starved_ms`` is the part of the gap in which this rank visited the
        loop with NOTHING to launch (fix 1r/2). Depth overlaps a rank's output
        exchange with its next forward; it cannot manufacture a chunk that the
        queue never supplied. Charging starvation to depth would buy in-flight
        slots against a supply problem -- and pay for them out of the KV pool.
        """
        d = self.denominator_ms
        if d <= 0.0:
            return 0.0
        return max(0.0, self.bubble_ms - self.starved_ms) / d


def read_pp_bubble(path: str) -> Optional[BubbleMeasurement]:
    """The BINDING stage's bubble totals from one group-P log, or None.

    The binding stage is the one with the largest forward total: under a
    pipelined prefill the makespan is that stage's time, so its idle is the
    idle that costs throughput. Returns None when the file does not exist or
    carries no PP-BUBBLE line -- absence of the instrument, never a measured
    zero (#892 / the indicator law: a tool says "I found nothing", never
    "there is nothing").
    """
    per_rank: Dict[int, List[float]] = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = _BUBBLE_RE.search(line)
                if m is None:
                    continue
                acc = per_rank.setdefault(int(m.group("rank")), [0.0] * 4 + [0.0])
                acc[0] += float(m.group("forward"))
                acc[1] += float(m.group("bubble"))
                acc[2] += float(m.group("starved"))
                acc[3] += float(m.group("n"))
                acc[4] += 1.0
    except OSError:
        return None
    if not per_rank:
        return None
    rank = max(per_rank, key=lambda r: per_rank[r][0])
    fwd, bub, starved, n_fwd, windows = per_rank[rank]
    return BubbleMeasurement(
        source=path,
        rank=rank,
        windows=int(windows),
        forward_ms=fwd,
        bubble_ms=bub,
        starved_ms=starved,
        n_forwards=int(n_fwd),
    )


def newest_bubble_log(evidence_dir: str) -> Optional[str]:
    """Newest ``*.P.log`` in ``evidence_dir`` that actually CARRIES the line.

    Not simply the newest P log: a boot that died before its first bubble
    window, or one built before the instrument existed, has no measurement,
    and taking its silence as ``share=0`` would derive ``depth=0`` from a file
    rather than from a measurement. Such a log is skipped and an older one
    that carries the line is preferred; the chosen path is printed, so a
    reader can see how old the number is.
    """
    try:
        names = [n for n in os.listdir(evidence_dir) if n.endswith(".P.log")]
    except OSError:
        return None
    paths = [os.path.join(evidence_dir, n) for n in names]
    for path in sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True):
        if read_pp_bubble(path) is not None:
            return path
    return None


#: The runtime's transport for an explicit per-stage layer SET. It is READ by
#: the ranks and WRITTEN by exactly one thing: this launcher, from
#: ``--pp-layer-set`` or from the solver's own chosen map. An INHERITED value
#: is refused (W44), never honoured -- see :func:`refuse_inherited_layer_set`.
PP_LAYER_SET_ENV = "SGLANG_PP_LAYER_SET"
PP_CROSSING_WIRE_ENV = "SGLANG_PP_CROSSING_WIRE"

_PREFILL_RE = re.compile(
    r"\b(?P<rank>PP\d+)\] Prefill batch,.*?#new-token: (?P<new>\d+), "
    r"#cached-token: (?P<cached>\d+)"
)


def refuse_inherited_layer_set(env: Mapping[str, str]) -> None:
    """W44: a layer set inherited from the environment is REFUSED, never used.

    ``build_env`` starts from ``os.environ``, so before this refusal an
    exported ``SGLANG_PP_LAYER_SET`` reached group P's ranks without passing
    through the launcher at all: the solver would rank and print one layout
    while the boot ran another, and the PP-CUT provenance line would describe
    a layout that never existed. That is the #505(a) silent-substitution class
    with a whole stage map as the substituted object.

    ONE KNOB, ONE READER (upstream-minimal). The flag ``--pp-layer-set`` is the
    only way to configure the map, and the launcher is the only writer of the
    variable. Not a twin of the env: its REPLACEMENT as the interface, with the
    variable demoted to the wire that carries the decision to the ranks.
    """
    raw = str(env.get(PP_LAYER_SET_ENV, "") or "").strip()
    if not raw:
        return
    raise Weg2LaunchRefused(
        "W44 Weg2LayerSetEnvRefused: %s=%r is set in this launcher's own "
        "environment. It is no longer an input: --pp-layer-set is, and the "
        "launcher is the only writer of the variable (it carries the solved "
        "or pinned map to group P's ranks). Honouring an inherited value "
        "would let the boot run a stage map the PP-CUT provenance line does "
        "not describe. Unset it and pass --pp-layer-set %s instead."
        % (PP_LAYER_SET_ENV, raw, raw)
    )


def read_mean_prefill_prefix(path: str) -> Optional[Tuple[float, int, str]]:
    """Mean PREFIX DEPTH over one boot's prefill census. ``(mean, n, rank)``.

    THE PREFIX IS ACCUMULATED, NOT READ. No line states it: a ``Prefill batch``
    line carries ``#new-token`` (this chunk) and ``#cached-token`` (the
    prefix-cache hit), never "how much of this request is already computed".
    But group P admits ONE chunked request per pass, so a request is a maximal
    run of consecutive lines on one rank and the prefix at chunk *j* is the sum
    of ``#new-token`` over chunks *0..j-1* plus that run's cache hit. A chunk
    SHORTER than the run's own maximum ends the request -- the last chunk of a
    prompt, or the zero-remainder 1-token end anchor -- and the accumulator
    resets.

    ONE RANK ONLY, and it is named in the answer: every stage prints the same
    batch, so summing across ranks would triple the denominator without adding
    a single measurement.

    ``None`` when the file does not exist or carries no census -- absence of
    the instrument, never a measured zero (#892).
    """
    rows: Dict[str, List[Tuple[int, int]]] = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = _PREFILL_RE.search(line)
                if m is None:
                    continue
                rows.setdefault(m.group("rank"), []).append(
                    (int(m.group("new")), int(m.group("cached")))
                )
    except OSError:
        return None
    if not rows:
        return None
    rank = sorted(rows)[0]
    seq = rows[rank]
    full = max(n for n, _ in seq)
    prefixes: List[int] = []
    acc = 0
    for new, cached in seq:
        prefixes.append(acc + cached)
        if new < full:
            acc = 0
        else:
            acc += new
    return (sum(prefixes) / float(len(prefixes)), len(prefixes), rank)


def newest_prefill_census_log(evidence_dir: str) -> Optional[str]:
    """Newest ``*.P.log`` that actually CARRIES a prefill census.

    Same rule and the same reason as :func:`newest_bubble_log`: a boot that
    died before its first prefill has no census, and reading its silence as
    "mean prefix 0" would derive the design depth from a file rather than from
    a measurement.
    """
    try:
        names = [n for n in os.listdir(evidence_dir) if n.endswith(".P.log")]
    except OSError:
        return None
    for path in sorted(
        (os.path.join(evidence_dir, n) for n in names),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    ):
        if read_mean_prefill_prefix(path) is not None:
            return path
    return None


def pcie_lanes(cards: Sequence[Card]) -> List[Optional[int]]:
    """Current PCIe link width per CUDA ordinal, from NVML. ``None`` if unknown.

    The measured link table this rig owns
    (``pp_crossing_transport.MEASURED_GBPS_BY_LANES``) is keyed by an edge's
    BOTTLENECK lane count, so the lane width is the join key -- read here per
    card rather than assumed, because "GPU0 is the x4 slot" is a fact about
    today's NVML enumeration and this launcher already refuses to hardcode
    that kind of fact (``order_cards``).
    """
    out: List[Optional[int]] = []
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return [None for _ in cards]
    try:
        for c in cards:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(int(c.nvml_index))
                out.append(int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)))
            except Exception:
                out.append(None)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def per_pair_crossing_ms(
    lanes: Sequence[Optional[int]], payload_bytes: int
) -> Dict[Tuple[int, int], float]:
    """``{(src_stage, dst_stage): ms}`` for one crossing of ``payload_bytes``.

    Priced from the MEASURED point-to-point table, at the edge's bottleneck
    lane count -- ``min`` of the two cards' widths, which is what "bottleneck"
    means and what those numbers were measured at. A pair whose lane count has
    no measured entry is OMITTED, never interpolated: ``pp_cut.crossing_price``
    then refuses to rank the candidates that use it, and an unrankable
    candidate is reported rather than priced at a guess.
    """
    from sglang.srt.distributed.pp_crossing_transport import MEASURED_GBPS_BY_LANES

    out: Dict[Tuple[int, int], float] = {}
    for a in range(len(lanes)):
        for b in range(len(lanes)):
            if a == b or lanes[a] is None or lanes[b] is None:
                continue
            gbps = MEASURED_GBPS_BY_LANES.get(min(int(lanes[a]), int(lanes[b])))
            if gbps is None:
                continue
            out[(a, b)] = float(payload_bytes) / (float(gbps) * 1e9) * 1e3
    return out


def chunked_prefill_size_of(argv: Sequence[str]) -> int:
    """The chunk this launcher's own argv states, read back off it.

    Same rule as ``--max-running-requests`` in the pool model: a second copy of
    the number here would drift the day the flag moves, and the depth price is
    linear in it.
    """
    argv = list(argv)
    if "--chunked-prefill-size" not in argv:
        raise Weg2LaunchRefused(
            "W42 Weg2DepthUnfunded: group P's argv states no "
            "--chunked-prefill-size, so one in-flight microbatch has no "
            "priceable size and the depth cannot be funded against the pool. "
            "Refusing rather than assuming a chunk."
        )
    return int(argv[argv.index("--chunked-prefill-size") + 1])


#: THE DERIVATION IS NOT RE-GROUNDED, so a derived depth is not shipped
#: (#1240 FOLLOW FIX 1, MUST_FIX 3 of BOOT_weg2pp2_0907.md). MEASURED against
#: this workload, cited to that record's DEPTH VERDICT table: depth 1 cost
#: +6.9 % on TTFT p50 (3.215 vs 3.008 s, A/A floor 1.7 %) and depth 2 +9.6 %
#: (3.298), the bubble share on the binding stage did NOT fall (21.8/24.6 % at
#: depth 0, 26.8 % at depth 1, 24.1 % at depth 2, floor 2.8 pts), duty fell 15
#: points, and at the ~60 k deep point all depths sat inside the 3.35 % floor
#: = null. The #692 mechanism overlaps a rank's output exchange with its next
#: forward; the measurement says this workload's gap is not that exchange --
#: consistent with the depth line's own stall_share=0.221 against
#: bubble_share=0.466, i.e. over half the bubble was already queue starvation,
#: which depth cannot touch, and the remainder is evidently not exchange-bound
#: either. Re-grounding means a MEASURED exchange-bound share to derive from;
#: until one exists the arithmetic is printed and the shipped depth is 0. An
#: explicit --p-microbatch-depth still wins, priced exactly as before.
DEPTH_DERIVATION_GROUNDED = False

#: The record line the retraction above is read from. Named so the number and
#: its provenance cannot drift apart.
DEPTH_RETRACTION_RECORD = "/spinning/gpu-arb/weg2/BOOT_weg2pp2_0907.md"


@dataclass(frozen=True)
class DepthDecision:
    """Group P's ``--pp-async-batch-depth`` and the ONE provenance line for it."""

    depth: int
    passes_in_flight: int
    pinned: bool
    measured: Optional[BubbleMeasurement]
    pool_tokens: float
    pool_after: float
    cap_tokens: int
    price_rows: int
    price_tokens: float
    price_mib: Tuple[float, ...]
    act_mib_per_pass: float
    chunk_tokens: int
    #: What the bubble ARITHMETIC asks for, before the retraction. Kept beside
    #: the shipped :attr:`depth` rather than replaced by it: the derivation is
    #: the thing that has to be re-grounded, so it stays visible and the day a
    #: measured exchange-bound share exists, ``DEPTH_DERIVATION_GROUNDED``
    #: flips and this number ships without a second derivation being written.
    derived_depth: int = 0
    #: True when the chosen layout is a GAPPED layer set. The depth is then 0
    #: BY THE LAYOUT, not lowered: a gapped map admits exactly one pass in
    #: flight, and that is the number the PP-CUT solver already priced the
    #: candidate at. Printed so the 0 is read as a derivation, not a default.
    gapped_layout: bool = False
    #: WHICH rule lowered the shipped depth below the derivation, recorded at
    #: the point it acts rather than inferred afterwards from the numbers:
    #: ``""`` (none acted -- the derivation shipped, or there was nothing to
    #: lower), ``"retraction"`` (the #692 derivation is not re-grounded) or
    #: ``"layout"`` (a gapped map admits exactly one pass). The two rules are
    #: branches of ONE ``elif`` in :func:`solve_p_depth`, so at most one can
    #: act, and :meth:`line` prints exactly one cause for a 0. Inferring it
    #: from ``derived_depth > depth`` cannot tell them apart, which is how the
    #: line came to print both.
    lowered_by: str = ""

    @property
    def retracted(self) -> bool:
        """True when the RETRACTION is the rule that lowered the depth."""
        return self.lowered_by == "retraction"

    @property
    def lowered_by_layout(self) -> bool:
        """True when the GAPPED LAYOUT is the rule that lowered the depth."""
        return self.lowered_by == "layout"

    def line(self) -> str:
        """THE one line. Format is load-bearing: a reader greps ``P-DEPTH solver:``."""
        if self.measured is None:
            src = (
                "no PP-BUBBLE measurement found, so bubble_share is unknown and "
                "the depth stays at today's behaviour"
            )
            shares = "bubble_share=n/a forward_share=n/a"
        else:
            m = self.measured
            src = (
                "from %s rank=%d (binding stage, largest forward total) over %d "
                "window(s), n=%d forwards; starved_ms=%.1f of bubble_ms=%.1f is "
                "queue starvation and is NOT charged to depth, stall_share=%.3f"
                % (
                    m.source,
                    m.rank,
                    m.windows,
                    m.n_forwards,
                    m.starved_ms,
                    m.bubble_ms,
                    m.stall_share,
                )
            )
            shares = "bubble_share=%.3f forward_share=%.3f" % (
                m.bubble_share,
                m.forward_share,
            )
        if self.retracted:
            src = (
                "the #692 DERIVATION IS RETRACTED and NOT re-grounded: it asks "
                "for depth %d, and %s (DEPTH VERDICT) measured that depth "
                "against this workload -- +6.9 %% on TTFT p50 at depth 1, "
                "+9.6 %% at depth 2, the bubble share did NOT fall (21.8/24.6 "
                "-> 26.8 -> 24.1 %%) and the deep point was inside its floor. "
                "The shipped depth is 0 until a MEASURED exchange-bound share "
                "exists to derive from; an explicit --p-microbatch-depth still "
                "wins and is still priced. " % (self.derived_depth, DEPTH_RETRACTION_RECORD)
                + src
            )
        if self.lowered_by_layout:
            src = (
                "the chosen layout is a GAPPED layer set, which admits exactly "
                "ONE pass in flight (scheduler_pp_mixin.init_pp_loop_state "
                "refuses the pair: under a gapped set each stage's next layer "
                "is another's previous one, so every stage must be inside the "
                "same forward). The depth is 0 BY THE LAYOUT -- the same one "
                "pass the PP-CUT solver priced that candidate at, which is why "
                "it is a derivation and not a lowered hand number. " + src
            )
        elif self.gapped_layout:
            # The bound is real and worth printing, but it did NOT act: the
            # depth was already at or below one pass. Saying "BY THE LAYOUT"
            # here would name a rule that never ran, which is the same
            # instrument-text defect as printing two causes for one 0.
            src = (
                "the chosen layout is a GAPPED layer set, which admits exactly "
                "ONE pass in flight; the depth was already 0, so the layout "
                "bound did not have to lower anything. " + src
            )
        return (
            "WEG2 P-DEPTH solver:%s%s %s depth=%d price_rows=%d/stage "
            "price_mib=%s MiB/stage pool_after=%d (constraint pool >= %d) "
            "[passes_in_flight=ceil(1/(1-stall_share))=%d, the flag is that "
            "minus the pass being forwarded; one extra pass costs %d KV rows "
            "plus a %.1f MiB crossing frame per stage, charged as %d pool "
            "tokens at the stage that converts worst; %s]"
            % (
                " PINNED (user override)" if self.pinned else "",
                " RETRACTED (derived %d, shipped %d)" % (self.derived_depth, self.depth)
                if self.retracted
                else "",
                shares,
                self.depth,
                self.price_rows,
                ",".join("%.1f" % v for v in self.price_mib),
                int(self.pool_after),
                int(self.cap_tokens),
                self.passes_in_flight,
                self.chunk_tokens,
                self.act_mib_per_pass,
                int(self.price_tokens),
                src,
            )
        )


def solve_p_depth(
    measured: Optional[BubbleMeasurement],
    pool_tokens: float,
    attn_counts: Sequence[int],
    kv_mib_per_token_per_attn_layer: float,
    hidden_size: int,
    chunk_tokens: int,
    cap_tokens: int,
    gapped_layer_set: str = "",
    dtype_bytes: int = 2,
    pinned_depth: Optional[int] = None,
) -> DepthDecision:
    """Group P's microbatch depth, DERIVED from the previous boot's own bubble.

    THE MECHANISM. ``init_pp_loop_state`` reads this knob twice: it widens
    ``pp_loop_size = pp_size + depth``, and -- the half that matters for the
    bubble -- ``_event_loop_pp_body`` moves
    ``_pp_commit_send_output_work_and_preprocess_output_tensors`` from AFTER
    ``_pp_launch_batch`` to BEFORE it when the depth is non-zero. At depth 0
    a rank's output exchange for pass *i-1* therefore serialises with its
    launch of pass *i*, and that serialisation is exactly the host time the
    PP-BUBBLE meter measures between two forwards.

    THE DERIVATION. A stage busy ``forward/(gap+forward)`` of the time needs
    ``ceil(1/(1-stall_share))`` passes in flight to stay busy across the gap.
    One of those passes is the one it is forwarding, so the FLAG -- which
    counts passes BEYOND the ring's own -- is that number minus one. Stated
    rather than folded in, because it is where this function departs from the
    briefing's ``depth = ceil(1/(1-bubble_share))``: that form provisions one
    extra in-flight pass beyond the gap it has to cover, and every extra pass
    is charged to the KV pool below. Both terms are printed, so a reader who
    wants the other convention can see the arithmetic rather than infer it.

    THE #692 GATE, RE-READ FOR WEG 2. ``DESIGN_691_bubble_levers.md`` gated
    this lever on "measure it against the seam, not just against throughput",
    for two named costs. They do not survive equally here:

    * *"more live KV on cards already failing seam funding at 179 MiB"* -- does
      NOT apply. That was the one-process flip, where the live set at the seam
      had to be FUNDED to survive a cutover. In Weg 2 group P is its own
      process group and carries nothing across: its sleep releases ``kv_cache``
      and the weight tags wholesale to the memory saver (``sleep_group``), and
      the carrier between the phases is the HiCache store, which D reads. There
      is no funded live set for depth to grow. What depth does cost is priced
      here instead, in the only budget it actually touches: P's own KV pool,
      against ``--max-kv-per-request``.
    * *"a deeper pipeline has more state to quiesce, so seam entry takes
      longer"* -- DOES apply, unchanged, and is not priced here. The front's
      quiesce witness is ``/flush_cache`` returning 200 only when every
      in-flight term is zero (``front.py`` witness B), so ``depth`` extra
      in-flight chunks are ``depth`` extra chunk-forwards of drain before P can
      sleep. At the measured 369.6 ms per full chunk on the binding stage that
      is sub-second per unit of depth against a 15-17 s flip, which is why it
      is named and left to the flip's own measurement rather than converted
      into a second, unmeasured budget here.

    Refuses W42 rather than lowering the depth: a silently reduced depth is a
    hand number wearing a derivation.
    """
    kv_mib_per_token = [
        max(1, int(a)) * float(kv_mib_per_token_per_attn_layer) for a in attn_counts
    ]
    # The activation the extra pass keeps alive: one PPProxyTensors
    # hidden-states frame per stage boundary, [chunk, hidden] in the model
    # dtype. It is NOT in pp_cut.PhasePoolModel (which prices weights, mamba
    # state, the arming floor and KV only), so it is the one genuinely new
    # term -- and it is converted into pool tokens rather than charged against
    # a second MiB budget, because the pool model has already spent every free
    # MiB into tokens and charging both would be two books for one byte.
    act_mib_per_pass = float(chunk_tokens) * float(hidden_size) * float(dtype_bytes)
    act_mib_per_pass /= 1024.0 * 1024.0

    if measured is None:
        passes = 1
    else:
        stall = measured.stall_share
        passes = int(math.ceil(1.0 / (1.0 - stall))) if stall < 1.0 else 0
    depth = max(0, passes - 1)
    derived_depth = depth
    pinned = pinned_depth is not None
    if pinned:
        if int(pinned_depth) < 0:
            raise Weg2LaunchRefused(
                "W42 Weg2DepthUnfunded: --p-microbatch-depth %d is negative; "
                "the flag counts in-flight passes." % int(pinned_depth)
            )
        # A pin replaces the derivation, never the PRICE: it is announced and
        # then funded on exactly the same axis, so an override cannot buy a
        # depth the pool cannot hold.
        depth = int(pinned_depth)
        passes = depth + 1

    gapped_layout = bool(gapped_layer_set)
    # ONE ZERO, ONE CAUSE. Both rules below produce depth 0, so they are
    # branches of one ``elif`` and the acting one is RECORDED: a reader
    # grepping ``P-DEPTH solver:`` for why the depth is 0 gets the rule that
    # ran, not two candidate explanations. The LAYOUT is tested first because
    # it is the structural bound -- a gapped map cannot run more than one pass
    # whatever the measurement later says -- while the retraction is a verdict
    # on the derivation that a re-grounded #692 will lift.
    lowered_by = ""
    if depth > 0 and gapped_layout and not pinned:
        # DERIVED, not lowered. The bubble measurement asks for more passes,
        # the layout admits one, and the PP-CUT solver already RANKED this
        # candidate at one pass (CutCandidate: a gapped map's stages do not
        # overlap, so its makespan was the SUM). Taking the depth the layout
        # admits is therefore consistent with the number that chose it; the
        # line says so rather than printing a bare 0.
        depth, passes = 0, 1
        lowered_by = "layout"
    elif depth > 0 and not pinned and not DEPTH_DERIVATION_GROUNDED:
        # RETRACTED BY MEASUREMENT, not lowered by taste. The arithmetic above
        # is unchanged and is printed; what does not ship is its OUTPUT, because
        # BOOT_weg2pp2_0907.md measured this knob against this workload and it
        # lost on every column that has a floor (see DEPTH_DERIVATION_GROUNDED).
        # A derivation whose premise the metal refuted is not a default, and
        # keeping it would be a hand number wearing a derivation just as much as
        # silently lowering one would be.
        depth, passes = 0, 1
        lowered_by = "retraction"

    # Only a PIN can still be non-zero against a gapped map: the derived path
    # was taken to 0 by the branch above.
    if depth > 0 and gapped_layout:
        raise Weg2LaunchRefused(
            "W43 Weg2DepthGapped: the layer set %r puts group P on a "
            "GAPPED layer set, and the PINNED --pp-async-batch-depth %d would "
            "die at scheduler_pp_mixin.init_pp_loop_state, which refuses the "
            "pair outright: under a gapped set every stage must be inside the "
            "SAME forward, because each stage's next layer is another's "
            "previous one, and async depth lets a rank enter the next forward "
            "while a peer is still in the last one's output exchange. Refused "
            "here so the launcher says it, rather than three ranks discovering "
            "it after the weights are loaded. A DERIVED depth is taken down to "
            "0 by the layout instead (the solver priced this candidate at one "
            "pass); an OVERRIDE is refused, because an override the boot cannot "
            "run is not a tuning choice." % (gapped_layer_set, depth)
        )

    price_rows = depth * int(chunk_tokens)
    price_mib = tuple(
        depth * (float(chunk_tokens) * k + act_mib_per_pass) for k in kv_mib_per_token
    )
    # The stage that converts worst is the one with the FEWEST attention
    # layers: a token costs it less MiB, so the same MiB of crossing frame
    # costs it MORE tokens. The pool is a MIN over stages, so that stage is
    # the one that binds it.
    price_tokens = (
        max(
            depth * (float(chunk_tokens) + act_mib_per_pass / k)
            for k in kv_mib_per_token
        )
        if depth > 0
        else 0.0
    )
    pool_after = float(pool_tokens) - price_tokens

    if depth > 0 and pool_after < float(cap_tokens):
        # WHO ASKED FOR THIS DEPTH decides the first and last sentence. While
        # DEPTH_DERIVATION_GROUNDED is False a derived depth is always 0, so
        # the only depth that can reach this floor today is a PIN -- and
        # opening with "the measured bubble asks for" then attributed the
        # operator's own override to a measurement, printed that measurement's
        # stall_share as if it had produced the number, and closed with a
        # sentence ("a depth the bubble did not ask for is a hand number")
        # that refutes itself on exactly this path. Both branches below are
        # live: the derived one the day #692 re-grounds.
        if pinned:
            asked = (
                "--p-microbatch-depth %d was PINNED (the measurement derived "
                "%d), and %d passes in flight do not fit group P's pool"
                % (depth, derived_depth, passes)
            )
            closing = (
                "The pin is NOT quietly lowered to what fits: an override "
                "that silently becomes another number is a hand number "
                "wearing a pin. Re-pin the depth the pool can hold (0 is a "
                "pin), or fund this one."
            )
        else:
            asked = (
                "the measured bubble asks for --pp-async-batch-depth %d (%d "
                "passes in flight at stall_share %.3f), but group P's pool "
                "cannot fund it"
                % (
                    depth,
                    passes,
                    0.0 if measured is None else measured.stall_share,
                )
            )
            closing = (
                "The depth is NOT quietly lowered to what fits: a depth the "
                "bubble did not ask for is a hand number. Pin depth 0 to "
                "accept today's behaviour."
            )
        raise Weg2LaunchRefused(
            "W42 Weg2DepthUnfunded: %s. Pool is %d tokens, one extra pass "
            "costs %d KV rows plus a %.1f MiB crossing frame per stage = %d "
            "pool tokens at the worst-converting stage, leaving %d against the "
            "%d-token floor (--max-kv-per-request) -- short by %d. Raise the "
            "per-rank budgets or lower --max-kv-per-request. %s"
            % (
                asked,
                int(pool_tokens),
                int(chunk_tokens),
                act_mib_per_pass,
                int(price_tokens),
                int(pool_after),
                int(cap_tokens),
                int(float(cap_tokens) - pool_after),
                closing,
            )
        )

    return DepthDecision(
        depth=depth,
        derived_depth=derived_depth,
        passes_in_flight=passes,
        pinned=pinned,
        measured=measured,
        pool_tokens=float(pool_tokens),
        pool_after=pool_after,
        cap_tokens=int(cap_tokens),
        price_rows=price_rows,
        price_tokens=price_tokens,
        price_mib=price_mib,
        act_mib_per_pass=act_mib_per_pass,
        chunk_tokens=int(chunk_tokens),
        gapped_layout=gapped_layout,
        lowered_by=lowered_by,
    )


def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


@dataclass(frozen=True)
class PCutFacts:
    """What the solved cut knows that the depth price also needs.

    Handed on rather than re-derived: the pool, the attention split and the KV
    cell are already solved once here, and a second derivation beside them
    would be a second set of books for one physical fact.
    """

    stage_ratio: str
    attn_stage_ratio: str
    pool_tokens: float
    attn_counts: Tuple[int, ...]
    kv_mib_per_token_per_attn_layer: float
    hidden_size: int
    cap_tokens: int
    #: The solver's chosen ownership map in ``--pp-layer-set`` syntax, and
    #: whether it is GAPPED. Empty string = a contiguous cut, which needs no
    #: map at all: the count form expresses it and every path below stays
    #: byte-identical to what it was.
    layer_set: str = ""
    gapped: bool = False


def solve_p_cut(
    ns,
    cards: List[Card],
    budgets_p: List[int],
    model: str,
    log,
    chunk_tokens: int = 4096,
) -> PCutFacts:
    """Group P's layer + attention cut, and the ONE provenance line for it.

    Returns the flag strings plus the pool facts the depth solver prices
    against (:class:`PCutFacts`). Everything it feeds the solver is either measured on this box
    (NVML card names, the per-rank budgets this launcher just derived, the
    checkpoint's own weight headers and KV cell) or a flag whose default
    carries its provenance in the help text -- no number is chosen here.
    """
    from sglang.srt.planner import pp_cut as _pp_cut
    from sglang.srt.planner import pp_cut_launch as _cut

    cfg_path = os.path.join(model, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    text_cfg = cfg.get("text_config") or cfg
    n_layers = int(text_cfg["num_hidden_layers"])
    kinds = text_cfg.get("layer_types") or []
    n_attn = sum(1 for k in kinds if str(k) == "full_attention")
    if n_attn <= 0:
        raise Weg2LaunchRefused(
            f"W40 Weg2PPCutRefused: {cfg_path} states no full_attention layer "
            f"in layer_types, so the attention axis cannot be solved and the "
            f"KV pool cannot be priced. Refusing rather than defaulting."
        )
    # The KV cell is CONSUMED from config, never fitted (#704 D1).
    kv_mib = _pp_cut.kv_mib_per_token_per_attn_layer_from_config(
        cfg, "fp8_e4m3", n_layers
    )
    # Weights from the safetensors HEADERS, not from a parameter formula.
    terms = _pp_cut.checkpoint_weight_terms(model)
    n_attn_ckpt = len(terms.attention_layer_indices)
    mean_layer_mib = (
        terms.attn_layer_weight_bytes * n_attn_ckpt
        + terms.linear_layer_weight_bytes * (terms.n_layers - n_attn_ckpt)
    ) / max(1, terms.n_layers) / _pp_cut.MIB
    ms = _csv_floats(ns.pp_cut_measured_ms_per_layer)
    model_pool = _pp_cut.PhasePoolModel(
        free_mib=tuple(float(b) for b in budgets_p),
        # The FAMILY split of weights is deliberately averaged: a stage's
        # divisor in stage_pp_capacities is its ATTENTION COUNT, and moving
        # one attention layer changes that divisor by 1 in 4-8 (12-25 %)
        # while changing weights by the attn/linear difference of a single
        # layer against a ~17-28 GiB free -- two orders of magnitude apart.
        # The total is exact for any cut summing to n_layers.
        weight_mib_per_layer=mean_layer_mib,
        kv_mib_per_token_per_attn_layer=kv_mib,
        arming_floor_mib=tuple(float(ns.pp_cut_arming_floor_mib) for _ in budgets_p),
        mamba_mib_per_linear_layer_per_slot=float(
            ns.pp_cut_mamba_mib_per_linear_layer_per_slot
        ),
        # Read off the argv this launcher builds rather than restated: a
        # second copy of --max-running-requests would drift the day the flag
        # moves, and the mamba residency scales linearly with it.
        mamba_slots=_max_running_requests(model),
    )
    families = tuple(
        _pp_cut.LAYER_FAMILY_ATTENTION
        if str(k) == "full_attention"
        else _pp_cut.LAYER_FAMILY_LINEAR
        for k in kinds
    )
    incumbent = _csv_ints(ns.pp_stage_ratio) if ns.pp_stage_ratio else _csv_ints("32,18,14")

    # -- THE DEPTH AXIS (#1240) ------------------------------------------
    # The design prefix is a MEASUREMENT of this rig's own traffic when one
    # exists, and a printed fallback when it does not. It is not cosmetic: the
    # attention/linear cost ratio moves by more than an order of magnitude
    # between 4,096 and 262,144 tokens of prefix, so "the optimal cut" is a
    # function of it and a table without it stated is a table about nothing.
    design_src = ns.pp_cut_design_prefix_from or newest_prefill_census_log(EVIDENCE_DIR)
    census = read_mean_prefill_prefix(design_src) if design_src else None
    if ns.pp_cut_design_prefix_tokens is not None:
        design_prefix = int(ns.pp_cut_design_prefix_tokens)
        design_prov = "PINNED by --pp-cut-design-prefix-tokens"
    elif census is not None:
        design_prefix = int(round(census[0]))
        design_prov = (
            "MEASURED mean prefix over %d prefill-batch chunks of %s (rank %s, "
            "one rank only: every stage prints the same batch)"
            % (census[1], design_src, census[2])
        )
    else:
        design_prefix = int(DESIGN_PREFIX_FALLBACK_TOKENS)
        design_prov = (
            "FALLBACK: no P log in %s carries a prefill census, so the design "
            "prefix is one chunk -- the shallowest depth this boot can run. "
            "This is an absence of the instrument, not a measured shallow rig."
            % EVIDENCE_DIR
        )

    measured_attn = _pp_cut.attention_counts(families, incumbent)
    anchor_stage = next(
        (i for i, c in enumerate(cards) if "5090" not in c.name), len(cards) - 1
    )
    family_cost, family_prov = _pp_cut.family_costs_from_measurement(
        measured_ms_per_layer=ms,
        measured_counts=incumbent,
        measured_attn_counts=measured_attn,
        chunk_tokens=int(chunk_tokens),
        ref_prefix_tokens=float(ns.pp_cut_calibration_prefix_tokens),
        anchor_stage=anchor_stage,
        anchor_attn_ms_per_layer=float(ns.pp_cut_attn_anchor_ms),
        anchor_prefix_tokens=float(ns.pp_cut_attn_anchor_prefix_tokens),
    )
    # THE CROSSING FRAME is the same object the depth price already charges:
    # one PPProxyTensors hidden-states frame, [chunk, hidden] in the model
    # dtype. Derived here from the same three numbers rather than restated.
    payload_bytes = int(chunk_tokens) * int(text_cfg["hidden_size"]) * 2
    lanes = pcie_lanes(cards)
    pair_ms = per_pair_crossing_ms(lanes, payload_bytes)
    log(
        "PP-CUT depth axis: design_prefix=%d tokens (%s); calibration prefix "
        "%d (%s); %s"
        % (
            design_prefix,
            design_prov,
            int(ns.pp_cut_calibration_prefix_tokens),
            "boot bsscale, ~12k prompt = 3 full chunks at 4096 -> mean 4096, "
            "BSSCALE_0907.md",
            family_prov,
        )
    )
    log(
        "PP-CUT crossing prices: PCIe lanes per ordinal %s, frame %.1f MiB "
        "(chunk %d x hidden %d x 2 B), measured pair ms %s (pairs absent from "
        "the measured lane table are OMITTED, and every candidate using one is "
        "reported UNPRICED rather than ranked at a guess)"
        % (
            lanes,
            payload_bytes / _pp_cut.MIB,
            int(chunk_tokens),
            int(text_cfg["hidden_size"]),
            ", ".join(
                "%d->%d %.2f" % (a, b, v) for (a, b), v in sorted(pair_ms.items())
            )
            or "NONE",
        )
    )
    decision = _cut.solve_launch_cut(
        layer_families=families,
        incumbent_layers=incumbent,
        measured_ms_per_layer=ms,
        family_cost=family_cost,
        design_prefix_tokens=design_prefix,
        per_pair_crossing_ms=pair_ms,
        pinned_layer_set=ns.pp_layer_set or None,
        measured_provenance=(
            "MEASURED per-layer ms %s (boot bsscale, BSSCALE_0907.md tip "
            "37c884b0b0: PP0 259.1 ms/32 layers, PP1 632.9/18, PP2 470.2/14, "
            "per full 4096-token chunk at bs6)" % ns.pp_cut_measured_ms_per_layer
        ),
        card_names=[c.name for c in cards],
        pool_model=model_pool,
        cap_tokens=int(ns.max_kv_per_request),
        pinned_layers=_csv_ints(ns.pp_stage_ratio) if ns.pp_stage_ratio else None,
        pinned_attn=_csv_ints(ns.pp_attn_stage_ratio) if ns.pp_attn_stage_ratio else None,
    )
    log(
        f"PP-CUT inputs: layers={n_layers} attn={n_attn} "
        f"kv={kv_mib * 1024 * 1024:.0f} B/token/attn-layer (from config, fp8_e4m3) "
        f"weights attn {terms.attn_layer_weight_bytes / _pp_cut.MIB:.1f} / linear "
        f"{terms.linear_layer_weight_bytes / _pp_cut.MIB:.1f} MiB per layer -> mean "
        f"{mean_layer_mib:.1f} used; free={budgets_p} MiB (this launcher's own "
        f"per-rank budgets); arming floor {ns.pp_cut_arming_floor_mib} MiB/rank; "
        f"mamba/linear-layer/slot {ns.pp_cut_mamba_mib_per_linear_layer_per_slot} MiB "
        f"(0.0 = UNFUNDED, pool is an UPPER bound)"
    )
    log(decision.provenance_line())
    for row in decision.table_lines():
        log(row)
    if decision.chosen.kind == "gapped":
        # A GAPPED map is not expressible as --pp-stage-ratio and does not go
        # through derive_pp_layer_split at all: it is published as the layer
        # SET and executed by the #753 mid-loop crossing wire. The round trip
        # below is therefore skipped by KIND, not by accident -- and the map is
        # instead round-tripped through the runtime's OWN parser, which is the
        # matching authority for this form.
        from sglang.srt.distributed.utils import parse_pp_layer_sets

        back = parse_pp_layer_sets(
            decision.chosen.layer_set, n_layers, len(budgets_p), allow_gapped=True
        )
        realized_counts = tuple(len(x) for x in back)
        if realized_counts != decision.chosen.layers:
            raise Weg2LaunchRefused(
                "W40 Weg2PPCutRefused: the gapped map this launcher solved "
                "(%s) does not survive parse_pp_layer_sets -- it comes back as "
                "%s layers per stage. Refusing rather than booting a layout the "
                "provenance line misdescribes."
                % (decision.chosen.layer_set, realized_counts)
            )
        log(
            "PP-CUT round trip: parse_pp_layer_sets(%s) = %s layers per stage, "
            "attn %s, %d crossings per chunk -- the --pp-layer-set argv states "
            "what the boot will run. --pp-stage-ratio/--pp-attn-stage-ratio are "
            "OMITTED from group P's argv for this kind: the count form cannot "
            "state a gapped map (derive_pp_layer_split refuses an attention "
            "count of 0 and silently re-derives a contiguous split from any "
            "other), so the map travels once, on the wire."
            % (
                decision.chosen.layer_set,
                realized_counts,
                ",".join(str(a) for a in decision.chosen.attn),
                decision.chosen.crossings,
            )
        )
        return PCutFacts(
            # EMPTY, DELIBERATELY (FOLLOW FIX 1). A gapped map's layout is the
            # SET, and the count form cannot state it: `48,8,8 / 0,8,8` is
            # refused by derive_pp_layer_split outright, and a positive-count
            # gapped map is accepted and derives a DIFFERENT contiguous split.
            # argv_p omits both flags when these are empty, so the argv states
            # exactly one layout -- the one on the wire.
            stage_ratio="",
            attn_stage_ratio="",
            pool_tokens=float(decision.chosen.pool_tokens),
            attn_counts=tuple(int(a) for a in decision.chosen.attn),
            kv_mib_per_token_per_attn_layer=float(kv_mib),
            hidden_size=int(text_cfg["hidden_size"]),
            cap_tokens=int(ns.max_kv_per_request),
            layer_set=decision.chosen.layer_set,
            gapped=True,
        )
    # ROUND TRIP AGAINST THE RUNTIME AUTHORITY, not against our own model.
    # --pp-stage-ratio entries are SCORES: server_args hands them to
    # derive_pp_layer_split, which SNAPS the boundary into the window that
    # realizes the attention target -- and snaps SILENTLY when an attention
    # vector is given (the #505(a) warning in _handle_pp_stage_ratio only
    # fires when attn_scores is None). MEASURED: 43,10,11 with attn 5,6,5
    # comes back as 23,24,17. A solved cut that does not survive this call is
    # a cut the boot would not run, so it is refused here rather than logged
    # and departed from.
    stage_ratio = ",".join(str(n) for n in decision.chosen.layers)
    attn_ratio = ",".join(str(a) for a in decision.chosen.attn)
    from sglang.srt.distributed.utils import derive_pp_layer_split

    realized = derive_pp_layer_split(
        list(decision.chosen.layers),
        is_full_attention=[str(k) == "full_attention" for k in kinds],
        attn_scores=list(decision.chosen.attn),
    )
    if list(realized) != list(decision.chosen.layers):
        raise Weg2LaunchRefused(
            f"W40 Weg2PPCutRefused: the cut this launcher solved "
            f"({stage_ratio} / attn {attn_ratio}) does not survive "
            f"derive_pp_layer_split -- it would run as "
            f"{','.join(str(c) for c in realized)}. Refusing rather than "
            f"booting a layout the provenance line misdescribes."
        )
    log(
        f"PP-CUT round trip: derive_pp_layer_split({stage_ratio}, attn "
        f"{attn_ratio}) = {','.join(str(c) for c in realized)} -- the argv "
        f"states what the boot will run."
    )
    return PCutFacts(
        stage_ratio=stage_ratio,
        attn_stage_ratio=attn_ratio,
        pool_tokens=float(decision.chosen.pool_tokens),
        attn_counts=tuple(int(a) for a in decision.chosen.attn),
        kv_mib_per_token_per_attn_layer=float(kv_mib),
        hidden_size=int(text_cfg["hidden_size"]),
        cap_tokens=int(ns.max_kv_per_request),
    )


def build_parser() -> argparse.ArgumentParser:
    """THE parser, built apart from :func:`main` so the desk can render it.

    ``argparse`` evaluates every ``help=`` string as ``help % params`` when it
    formats it, so one bare ``%`` in one flag's help makes ``--help`` raise for
    ALL of them. That is not a cosmetic failure: the boot record cites group
    D's measured default as readable "verbatim from its own help path", and a
    channel that cannot be opened is not provenance. Splitting the build out is
    what lets a unit test call ``format_help()`` once per commit instead of an
    operator discovering it in a terminal.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--venv", default=VENV_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug-hold", choices=["none", "P", "D", "both"], default="none")
    ap.add_argument("--store-min-gib", type=float, default=8.0)
    ap.add_argument("--weight-chunks", type=int, default=8,
                    help="#1233: number of weights_<k> layer-chunk tags per group (0 = the round-1 single tag / two-backup shape)")
    ap.add_argument("--ready-deadline-s", type=float, default=900.0)
    ap.add_argument("--extra-p", default="", help="extra flags for group P (shell-split)")
    ap.add_argument("--extra-d", default="", help="extra flags for group D (shell-split)")
    ap.add_argument("--fairness-w-s", type=float, default=45.0)
    ap.add_argument(
        "--transport", choices=["bar1", "nccl"], default="bar1",
        help="Collective transport for BOTH groups. 'bar1' is the shipping "
             "default. 'nccl' is the DEVELOPMENT mode of the user's order of "
             "2026-09-07: it drops the barlink flag family so the stock "
             "sglang dispatch builds a PyNccl communicator, and it raises "
             "group D's dormant-residue slack 64 -> 192 MiB because NCCL's "
             "buffers are not memory-saver-tagged and survive the sleep "
             "(measured 2230 -> 2310 MiB on the 5090, boot weg2ab0). Under "
             "'bar1' nothing about the flip path changes.",
    )
    # -- group P: the layer cut ------------------------------------------
    ap.add_argument(
        "--pp-stage-ratio", default=None,
        help="OVERRIDE the solved layer cut for group P (e.g. '32,18,14'). "
             "Passing it is announced as 'PINNED (user override)' in the "
             "PP-CUT provenance line and priced on the same two axes as the "
             "solved cut. Unset = the solver decides.",
    )
    ap.add_argument(
        "--pp-attn-stage-ratio", default=None,
        help="OVERRIDE the solved full-attention split for group P (e.g. "
             "'8,4,4'). Same PINNED provenance. Unset = the attention axis is "
             "resolved for the chosen layer cut by pp_cut.best_attention_split.",
    )
    ap.add_argument(
        "--max-kv-per-request", type=int, default=CONTEXT_LENGTH,
        help=f"Group P's KV pool must hold at least this many tokens: one "
             f"whole prompt must be device-resident for the stage's attention "
             f"layers while its prefill runs. Default {CONTEXT_LENGTH} = "
             f"--context-length, i.e. the longest prompt this boot admits. "
             f"A cut below it is refused (W40), never silently taken.",
    )
    ap.add_argument(
        "--pp-cut-measured-ms-per-layer", default=MEASURED_MS_PER_LAYER,
        help=f"Per-stage prefill ms per layer for the cost model. Default "
             f"'{MEASURED_MS_PER_LAYER}' is MEASURED on boot bsscale "
             f"(/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip 37c884b0b0, "
             f"per full 4096-token chunk at bs6: PP0 259.1 ms/32 layers, PP1 "
             f"632.9/18, PP2 470.2/14). Used as the ANCHOR when the measured "
             f"card-rate library supplies the per-card ratios.",
    )
    ap.add_argument(
        "--pp-cut-arming-floor-mib", type=float, default=ARMING_FLOOR_MIB,
        help=f"Per-rank arming floor subtracted before KV in the pool model. "
             f"Default {ARMING_FLOOR_MIB} = the top of the rig's VRAM "
             f"corridor (819-1229 MiB NVML-free per card under load). "
             f"pp_cut.PhasePoolModel names the gap this stands in for: the "
             f"solved per-layout floor only exists for a layout that has "
             f"booted, and the proxy carries about +-500 MiB.",
    )
    ap.add_argument(
        "--p-microbatch-depth", type=int, default=None,
        help="OVERRIDE group P's solved --pp-async-batch-depth (#692). Unset = "
             "DERIVED from the previous boot's own PP-BUBBLE line: a stage busy "
             "forward/(gap+forward) of the time needs ceil(1/(1-stall_share)) "
             "passes in flight to cover its gap, and the flag counts the passes "
             "BEYOND the one being forwarded. Priced against P's KV pool -- one "
             "extra pass costs one chunk of KV rows plus one crossing "
             "activation frame per stage -- and refused as W42 rather than "
             "lowered when the pool cannot fund it. Passing it is announced as "
             "PINNED and priced on the same axis.",
    )
    ap.add_argument(
        "--p-bubble-measured-from", default="",
        help="Path of the group-P log whose PP-BUBBLE lines feed "
             f"--p-microbatch-depth. Unset = the newest log in {EVIDENCE_DIR} "
             "that actually carries the instrument (a log without the line is "
             "skipped, never read as a measured zero). No measurement anywhere "
             "= depth 0, i.e. today's behaviour, and the line says so.",
    )
    ap.add_argument(
        "--pp-layer-set", default=None,
        help="OVERRIDE the solved stage map with an explicit per-stage LAYER "
             "SET, e.g. '0-2,4-6,...,60-62;3,7,...,31;35,...,63' (stages "
             "separated by ';'). THE ONLY WAY to configure the map: an "
             "inherited SGLANG_PP_LAYER_SET in the launcher's environment is "
             "refused by name (W44), never honoured, because the launcher is "
             "the only writer of that variable and the solver's provenance "
             "line must describe the layout the boot actually runs. Unset = "
             "the solver enumerates gapped maps alongside contiguous cuts and "
             "publishes its own choice. Passing it is announced as PINNED and "
             "priced on the same axes.",
    )
    ap.add_argument(
        "--pp-cut-design-prefix-tokens", type=int, default=None,
        help="OVERRIDE the prefix depth the layout is optimised FOR. Unset = "
             "DERIVED from the newest P log carrying a prefill census (the "
             "mean of the accumulated per-chunk prefix on one rank); no census "
             f"anywhere = {DESIGN_PREFIX_FALLBACK_TOKENS}, and the line says "
             "FALLBACK. It matters because a full-attention layer's cost per "
             "chunk grows with the prefix while a GDN layer's does not, so the "
             "optimal placement of the 16 attention layers is a FUNCTION of "
             "this number, not a constant.",
    )
    ap.add_argument(
        "--pp-cut-design-prefix-from", default="",
        help="Path of the P log whose prefill census feeds "
             f"--pp-cut-design-prefix-tokens. Unset = the newest log in "
             f"{EVIDENCE_DIR} that actually carries the census (a log without "
             "it is skipped, never read as a measured prefix of 0).",
    )
    ap.add_argument(
        "--pp-cut-calibration-prefix-tokens", type=float,
        default=float(CALIBRATION_PREFIX_TOKENS),
        help=f"The prefix depth --pp-cut-measured-ms-per-layer was MEASURED at. "
             f"Default {CALIBRATION_PREFIX_TOKENS} is DERIVED: boot bsscale "
             f"drove a ~12,000-token prompt at chunk 4096, i.e. three full "
             f"chunks entering at prefix 0/4096/8192, whose mean is 4096 "
             f"(BSSCALE_0907.md). Half of the two-depth family split.",
    )
    ap.add_argument(
        "--pp-cut-attn-anchor-ms", type=float, default=ATTN_ANCHOR_MS,
        help=f"MEASURED cost of ONE full-attention layer per chunk at the deep "
             f"anchor prefix, ms. Default {ATTN_ANCHOR_MS} is the user's "
             f"physics note of 2026-09-07, recorded in the GAPPED block of "
             f"WEG2_BUILD_DECISIONS_0906.md section 1r. The other half of the "
             f"two-depth split: one measurement cannot separate a stage's "
             f"attention cost from its GDN cost, two at different depths can.",
    )
    ap.add_argument(
        "--pp-cut-attn-anchor-prefix-tokens", type=float,
        default=float(ATTN_ANCHOR_PREFIX_TOKENS),
        help=f"The prefix the deep anchor was measured at. Default "
             f"{ATTN_ANCHOR_PREFIX_TOKENS} = --context-length.",
    )
    ap.add_argument(
        "--pp-cut-mamba-mib-per-linear-layer-per-slot", type=float, default=0.0,
        help="Device GDN state per linear layer per sequence slot, MiB. "
             "Default 0.0 = UNFUNDED, and the direction is named rather than "
             "hidden: omitting it inflates every stage's capacity, more for "
             "stages holding more linear layers, so the printed pool is an "
             "UPPER bound and the pool floor is looser than reality. Pass a "
             "measured value to tighten it.",
    )
    # -- group D: the decode knobs ---------------------------------------
    ap.add_argument(
        "--num-continuous-decode-steps", type=int,
        default=D_NUM_CONTINUOUS_DECODE_STEPS,
        help=f"Group D only. Decode steps run per scheduler visit "
             f"(server_args.py:1424). Default "
             f"{D_NUM_CONTINUOUS_DECODE_STEPS} is MEASURED, not chosen: arms "
             f"table row D2 of /spinning/gpu-arb/weg2/BOOT_weg2pp1_0907.md "
             f"reads 75.5 tok/s at bs1 and 305.0 at bs6 against the D0 "
             f"control's 66.2 / 284.9 (+14.0 %% / +7.1 %%), and row D3 shows 4 "
             f"REGRESSES to 73.6 / 295.6. Pass 1 to restore the shipped value.",
    )
    ap.add_argument(
        "--d-disable-overlap-schedule", action="store_true",
        help="Put --disable-overlap-schedule back on group D. The escape "
             "hatch for a server_args gate that refuses overlap under D's "
             "combination; taking it logs W41 with the gate named. Not a "
             "tuning knob.",
    )
    # -- measurement arms -------------------------------------------------
    ap.add_argument(
        "--p-hicache-write-policy", default="write_through",
        choices=["write_through", "write_back", "write_through_selective"],
        help="MEASUREMENT ONLY. Group P's hicache write policy. Default "
             "write_through = unchanged shipped behaviour. 'off' is NOT "
             "offered because this runtime has no such policy "
             "(server_args.py:4318 choices); write_back is the nearest arm "
             "-- it defers the store write rather than removing it, and the "
             "structural removal (a shared ring, async ack) belongs to the "
             "ring slice, not here.",
    )
    ap.add_argument("--teardown", default="", help="path of a boot state json to tear down")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.teardown:
        return teardown(ns.teardown)
    # BEFORE build_env(), which starts from os.environ: an inherited stage map
    # would reach group P's ranks without passing through the solver at all.
    refuse_inherited_layer_set(os.environ)

    tree = os.path.abspath(ns.tree)
    tip = subprocess.run(["git", "-C", tree, "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", tree, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    stamp = time.strftime("%m%d_%H%M%S", time.gmtime())
    base = f"{EVIDENCE_DIR}/boot_weg2_{ns.tag}_{tip}_{stamp}"
    dry = ns.dry_run
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    front_log = None if dry else f"{base}.front.log"
    log = Log(front_log)
    log(f"=== WEG2 BOOT tag={ns.tag} tree={tree} @ {tip} ({'DIRTY: ' + dirty[:200] if dirty else 'clean'}) stamp={stamp} dry={dry}")
    if dirty and not dry:
        raise Weg2LaunchRefused("tree is not clean -- boot from a COMMITTED tip only")
    py = f"{ns.venv}/bin/python"
    state = BootState(tag=ns.tag, tip=tip, tree=tree, stamp=stamp)
    state.logs = {"front": front_log or "<dry>", "P": f"{base}.P.log", "D": f"{base}.D.log"}

    # 1. preflight
    presence_sweep(log, ns.tag, stamp, dry)
    stale_deadman_sweep(log, [PORT_FRONT, PORT_P, PORT_D], dry)
    host_preflight(log, ns.tag, dry)
    cards = order_cards(resolve_cards())
    state.cards = [c.__dict__ for c in cards]
    if not dry:
        cards_free_check(cards, log)
    cvd = ",".join(c.uuid for c in cards)
    state.cvd = cvd
    log("NVML -> CUDA ordinal map: " + ", ".join(f"ordinal {i} = nvml {c.nvml_index} {c.name} {c.uuid} total {c.total_mib} MiB" for i, c in enumerate(cards)))

    # 1b. #1233 one-backup flip geometry + the patched saver hook
    n_layers = model_num_layers(ns.model)
    chunk_count = max(0, int(ns.weight_chunks))
    chunk_layers = int(math.ceil(n_layers / chunk_count)) if chunk_count > 0 else 0
    from sglang.srt.managers.weg2_memory_saver import weights_family_tags
    weights_tags = weights_family_tags(chunk_count)
    log(f"WEG2-WEIGHT-CHUNKS N={chunk_count} tags (layers per chunk {chunk_layers} of {n_layers}; family {weights_tags}); "
        "flip = src.pause(kv) -> per tag: src.pause(w_k), dst.resume(w_k) -> dst.resume(kv); host holds one image + one chunk")
    tms_so = "" if dry else build_tms_preload(tree, ns.venv, log)
    state.weight_chunks = chunk_count
    state.tms_so = tms_so

    # 2. host ledger
    mi = host_ledger.read_meminfo()
    arm, store_gib, lines = host_ledger.choose(mi["MemTotal"], mi["MemAvailable"], store_min_gib=ns.store_min_gib, weight_chunks=chunk_count)
    for ln in lines:
        log(ln)
    state.ledger_lines = lines
    state.store_gib = store_gib

    # 2b. host memory time series from BEFORE the first group, so the launch
    # moment (P image resident, D loading) is measured this time, not only
    # the run moment (record 1h: the ls1b2 sampler started after D READY).
    if not dry:
        memts = f"{GPU_ARB}/memts_weg2_{ns.tag}.csv"
        mpid = int(subprocess.run(["bash", "-c", f"setsid {MEMTS} {shlex.quote(memts)} 5 > /dev/null 2>&1 & echo $!"], capture_output=True, text=True).stdout.strip() or 0)
        state.helper_pids.append(mpid)
        log(f"mem time series pid {mpid} csv {memts} (started before group P: launch AND run moments sampled)")

    # 3. store
    mount_store(log, store_gib, dry)
    store_dir = f"{STORE_MOUNT}/store"
    if not dry:
        os.makedirs(store_dir, exist_ok=True)

    # 4. group P
    slack_mib = reserve_slack_mib(ns.transport)
    dc_expect_d = {
        c.uuid: (DC_MEASURED_D_5090_MIB if "5090" in c.name else DC_MEASURED_D_3080_MIB) + slack_mib
        for c in cards
    }
    if ns.transport == "nccl":
        log(
            f"TRANSPORT=nccl (development mode, user order 2026-09-07): barlink flags dropped from both groups; "
            f"group D's dormant-residue slack raised {DC_RESERVE_SLACK_MIB} -> {slack_mib} MiB because libnccl's buffers "
            f"are not memory-saver-tagged and survive the sleep (MEASURED 2230 -> 2310 MiB on the 5090, boot weg2ab0 ARM 0, "
            f"where the unchanged 2228+64 reserve tripped W19 DormantResidueRefused). Under --transport bar1 this number "
            f"and the whole flip path are untouched."
        )
    state.dc_expect_d = dc_expect_d
    log("dormant residue RESERVE for group D = MEASURED D_c(D) of boot weg2ls1b2 (2228 / 1922 / 1922 MiB, "
        f"NVML per-process, windows included) + {slack_mib} MiB slack; spec 1.6 expectation was "
        f"{DC_EXPECT_5090_MIB}/{DC_EXPECT_3080_MIB} (exceeded); graded by W19 at D's first sleep: "
        + ", ".join(f"nvml{c.nvml_index}={dc_expect_d[c.uuid]}" for c in cards))
    budgets_p = budgets_from_dc(
        cards, dc_expect_d, log, "P", overshoot_mib=P_OVERSHOOT_MIB, overshoot_provenance="boot weg2ls2b2"
    )
    state.budgets["P"] = budgets_p
    env_p = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("P", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport)
    # #1233 zero-remainder: group P ends every prefill's last chunk at N-1 and
    # publishes the recurrent anchor there (schedule_policy END-OF-PREFILL
    # ANCHOR); D can claim at most N-1 tokens of a prompt, so this is the
    # anchor it resumes from. P only: D's finish anchors serve the NEXT turn.
    env_p["SGLANG_WEG2_END_ANCHOR"] = "1"
    chunk_tokens = chunked_prefill_size_of(
        common_flags(ns.model, arm.s_gb, arm.m_mib, store_gib, ns.p_hicache_write_policy)
    )
    cut = solve_p_cut(ns, cards, budgets_p, ns.model, log, chunk_tokens=chunk_tokens)
    stage_ratio, attn_stage_ratio = cut.stage_ratio, cut.attn_stage_ratio
    # #1240 THE LAUNCHER IS THE ONLY WRITER. The solved (or pinned) map is
    # published here, into the environment group P will actually get -- the
    # flag is the interface, the variable is the wire. A GAPPED map also arms
    # the #753 mid-loop crossing wire, because that is the executor: without
    # it a stage runs its own layers back to back and silently skips its
    # peer's, and get_pp_layer_set refuses a gapped set with the wire off.
    if cut.layer_set:
        env_p[PP_LAYER_SET_ENV] = cut.layer_set
        if cut.gapped:
            env_p[PP_CROSSING_WIRE_ENV] = "1"
        log(
            "WEG2 LAYER-SET published for group P: %s=%s%s (source: %s). The "
            "#753 mid-loop crossing wire is the executor; nothing is "
            "duplicated here."
            % (
                PP_LAYER_SET_ENV,
                cut.layer_set,
                " " + PP_CROSSING_WIRE_ENV + "=1" if cut.gapped else "",
                "--pp-layer-set (PINNED)" if ns.pp_layer_set else "the solver",
            )
        )
    # #692 MICROBATCH DEPTH. The share is READ from a previous boot's own
    # PP-BUBBLE line, never written here: an operator-supplied path wins, else
    # the newest log in EVIDENCE_DIR that actually carries the instrument, else
    # no measurement at all and the depth is today's 0 -- printed, so the
    # absence is visible rather than inferred from a missing line.
    bubble_src = ns.p_bubble_measured_from or newest_bubble_log(EVIDENCE_DIR)
    if ns.p_bubble_measured_from and read_pp_bubble(ns.p_bubble_measured_from) is None:
        raise Weg2LaunchRefused(
            f"W42 Weg2DepthUnfunded: --p-bubble-measured-from "
            f"{ns.p_bubble_measured_from} carries no PP-BUBBLE line, so it "
            f"states nothing about the bubble. An explicitly named measurement "
            f"file that turns out to be empty is refused rather than silently "
            f"treated as 'no measurement' -- that would read as depth 0 for a "
            f"reason the operator did not intend."
        )
    depth_decision = solve_p_depth(
        measured=read_pp_bubble(bubble_src) if bubble_src else None,
        pool_tokens=cut.pool_tokens,
        attn_counts=cut.attn_counts,
        kv_mib_per_token_per_attn_layer=cut.kv_mib_per_token_per_attn_layer,
        hidden_size=cut.hidden_size,
        chunk_tokens=chunk_tokens,
        cap_tokens=cut.cap_tokens,
        # Read from the environment group P will actually get -- which since
        # #1240 this launcher is the only writer of (W44 refuses an inherited
        # one), so it is the solver's own decision arriving here rather than
        # an operator's export sneaking past the flags.
        gapped_layer_set=env_p.get(PP_LAYER_SET_ENV, "") if cut.gapped else "",
        pinned_depth=ns.p_microbatch_depth,
    )
    log(depth_decision.line())
    state.p_depth = depth_decision.depth
    if ns.d_disable_overlap_schedule:
        log(
            "W41 Weg2OverlapRefused: --d-disable-overlap-schedule was passed, "
            "so group D keeps --disable-overlap-schedule. The gate that "
            "refused must be named in this boot's record; the flag is not a "
            "tuning knob and no server_args gate is to be weakened to avoid it. "
            + d_overlap_cost_line(ns.model, True)
        )
    else:
        log(
            "SCHEDULER: --disable-overlap-schedule is group P's flag only "
            "(pp_size=3, server_args.py:19507). Group D runs pp_size=1 with "
            "the overlap scheduler ON; every forcing gate was checked against "
            "D's argv (pdmux off, device cuda, no sparse-head/dllm, and the "
            "hybrid-mamba resolution picks extra_buffer for "
            "Qwen3_5ForConditionalGeneration on linear_attn_backend=triton). "
            f"--num-continuous-decode-steps {ns.num_continuous_decode_steps} "
            f"on D (default {D_NUM_CONTINUOUS_DECODE_STEPS} is MEASURED, cited "
            f"to /spinning/gpu-arb/weg2/BOOT_weg2pp1_0907.md arms table row "
            f"D2: 75.5 tok/s bs1 and 305.0 bs6 against the D0 control's 66.2 / "
            f"284.9 = +14.0 % / +7.1 %, with row D3 showing steps 4 regresses "
            f"to 73.6 / 295.6 -- an optimum, not a direction). "
            + d_overlap_cost_line(ns.model, False)
        )
    if ns.p_hicache_write_policy != "write_through":
        log(
            f"MEASUREMENT ARM: group P --hicache-write-policy "
            f"{ns.p_hicache_write_policy} (shipped default is write_through; "
            f"#1016 measured the write_through store tax at +3.9 % @50k / "
            f"+11.6 % @12k and BSSCALE_0907.md P5 did NOT re-A/B it -- this "
            f"arm exists to, and changes nothing else). Group D is unchanged."
        )
    spec_p = GroupSpec("P", PORT_P, transport_argv(argv_p(py, ns.model, budgets_p, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_p), stage_ratio, attn_stage_ratio, ns.p_hicache_write_policy, depth_decision.depth), ns.transport), state.logs["P"], env_p)
    state.argv["P"] = " ".join(shlex.quote(a) for a in spec_p.argv)
    state.deviations = [
        "transports stay OPEN across sleep (barlink_reopen() unwired this round; BAR1 windows sized to fit both groups: P 24+96, D 16+32+40 MiB "
        "= 208 of 224 usable, measured Used 224/256 incl. RM carve-out; #1234 C1 raised dcp:0 from 24 so the 96-MiB dcp all_reduce plans to 10 rounds "
        "instead of 17 and stops falling to the host-staged plane)",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 on both groups (K2); /health_generate only via the front to the awake group",
        "group deadmen tier 1 only (PROBE_S spaced past the boot); tier 2 on the front's deadman",
        "no disk tier for the store (file backend has no second tier); tmpfs sized by the ledger",
        "the epoch is carried in the front's ledger only (no group echo/checksum on the RPC)",
        "torch_memory_saver 0.0.9.post1 preload hook rebuilt from the vendored csrc with ONE patch (cpu backup freed after resume, "
        "python/sglang/srt/weg2/tms_csrc/PATCH.md); the stock wheel is untouched and used when SGLANG_WEG2_TMS_PRELOAD_SO is unset",
        f"weights paused/resumed as {chunk_count} weights_<k> chunk tags + the base tag (#1233 one-backup flip); cuda_graph stays resident (not in the sleep tag set, as in round 1)",
        "zero-remainder: group P holds the last token of every prefill back into its own chunk (SGLANG_WEG2_END_ANCHOR=1) so the GDN anchor D resumes from sits at N-1; "
        "costs P one 1-token pass per request and serialises whole-fit prompts behind the one chunked request per pass",
        "zero-remainder: /flush_cache publishes un-backed nodes before the idle witness on both groups (SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP=1); the quiesce then waits for those backups",
        "zero-remainder: a BATCH prompt longer than group D's host staging pool can carry (launcher-measured from D's log, x0.9 prefetch bound) is served by ONE prefill on D "
        "(front route CARRIER-EXCEEDS, no leg 1) -- served and single-prefill, but NOT a zero-remainder leg 2; the windowed prefetch that would lift it is the next round",
        "zero-remainder: BATCH streams are priced post hoc via stream_options.include_usage (one standard trailing usage chunk reaches the client)",
    ]
    for d in state.deviations:
        log(f"DEVIATION (declared): {d}")
    launch_group(spec_p, tree, log, dry)
    if dry:
        budgets_d = budgets_from_dc(cards, {c.uuid: dc_expect_d[c.uuid] + P_WINDOWS_MIB - D_WINDOWS_MIB for c in cards}, log, "D(dry, expectation)")
        env_d = build_env(tree, ns.venv, cvd, store_dir, False, ns.tag, chunk_layers, chunk_count, tms_so, ns.transport)
        spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d), ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule), ns.transport), state.logs["D"], env_d)
        launch_group(spec_d, tree, log, dry)
        log("DRY-RUN complete: nothing started, mounted, armed or written")
        return 0
    state.pids["P"] = spec_p.pid
    _write_state(state)
    state.t_ready["P"] = wait_ready(PORT_P, spec_p.pid, ns.ready_deadline_s, log, "P", spec_p.proc)
    n_kv = count_marker(spec_p.log, "#706 canonical KV page active")
    n_blob = count_marker(spec_p.log, "canonical GDN blob active")
    log(f"W7/W10 launcher half, group P log: '#706 canonical KV page active' x{n_kv}, 'canonical GDN blob active' x{n_blob} (need >= 3 each: three ranks)")
    if n_kv < 3 or n_blob < 3:
        raise Weg2LaunchRefused(f"W7 Weg2MambaBlobAbsent / W10 Weg2CanonicalPageMissing (launcher half): P logged kv x{n_kv} blob x{n_blob}, need 3 each")

    # 4d. sleep P, measure D_c(P)
    state.sleep_p_ms = sleep_group(PORT_P, log, "P", weights_tags)
    time.sleep(2)
    pids_p = session_pids(spec_p.pid)
    dc_p = nvml_process_mib(pids_p)
    for c in cards:
        dc_p.setdefault(c.uuid, 0)
    uf = nvml_used_free(cards)
    for c in cards:
        exp = DC_EXPECT_5090_MIB if "5090" in c.name else DC_EXPECT_3080_MIB
        log(
            f"WEG2-DC group=P nvml{c.nvml_index} {c.name}: measured {dc_p[c.uuid]} MiB per-process "
            f"(pids {sorted(pids_p)}) card used {uf[c.uuid][0]} MiB; expectation {exp} + P windows {P_WINDOWS_MIB} = {exp + P_WINDOWS_MIB} MiB "
            f"({'AT OR BELOW' if dc_p[c.uuid] <= exp + P_WINDOWS_MIB else 'ABOVE'} expectation; the launcher derives D from the MEASUREMENT, record 1f B6)"
        )
    state.dc_measured_p = dc_p

    # 5. group D
    budgets_d = budgets_from_dc(
        cards, dc_p, log, "D", overshoot_mib=D_OVERSHOOT_MIB, overshoot_provenance="boot weg2ls4b1"
    )
    state.budgets["D"] = budgets_d
    env_d = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("D", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport)
    spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d), ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule), ns.transport), state.logs["D"], env_d)
    state.argv["D"] = " ".join(shlex.quote(a) for a in spec_d.argv)
    launch_group(spec_d, tree, log, dry)
    state.pids["D"] = spec_d.pid
    _write_state(state)
    state.t_ready["D"] = wait_ready(PORT_D, spec_d.pid, ns.ready_deadline_s, log, "D", spec_d.proc)
    n_kv = count_marker(spec_d.log, "#706 canonical KV page active")
    n_blob = count_marker(spec_d.log, "canonical GDN blob active")
    log(f"W7/W10 launcher half, group D log: '#706 canonical KV page active' x{n_kv}, 'canonical GDN blob active' x{n_blob}")
    if n_kv < 3 or n_blob < 3:
        raise Weg2LaunchRefused(f"W7/W10 (launcher half): D logged kv x{n_kv} blob x{n_blob}, need 3 each")
    # #1233 zero-remainder (1j finding 6): W9 LAUNCH-TIME KEY-SCHEME GATE. The
    # store is one carrier; a spec-less group keys pages by unigram unless
    # SGLANG_HICACHE_BIGRAM_KEYS=1 forced the bigram scheme, a NEXTN/EAGLE
    # group keys by bigram natively. Divergent schemes wrote disjoint chains
    # for the same prompt once (boots weg2ls3b1-b3); the identity-suffix
    # check in the front cannot see it. Refuse the boot before any traffic.
    _forced = "#1233 HICACHE BIGRAM KEYS FORCED"
    _p_bigram = ("--speculative-algorithm" in spec_p.argv) or count_marker(spec_p.log, _forced) >= 1
    _d_bigram = ("--speculative-algorithm" in spec_d.argv) or count_marker(spec_d.log, _forced) >= 1
    log(f"W9 launch-time key-scheme gate: P bigram={_p_bigram} (forced lines {count_marker(spec_p.log, _forced)}) "
        f"D bigram={_d_bigram} (forced lines {count_marker(spec_d.log, _forced)})")
    if _p_bigram != _d_bigram:
        raise Weg2LaunchRefused(f"W9 Weg2StoreIdentityMismatch (launch-time key scheme): P bigram={_p_bigram} D bigram={_d_bigram} -- "
                                f"the two groups would key the sole carrier by different page-hash schemes")
    # #1233 zero-remainder: the carrier bound the front routes by -- group D's
    # smallest host staging pool (tokens) x 0.9 (the prefetch rate bound that
    # refused the 84k prompt on boot weg2ls4b2: limit=27466 of pool 30518).
    _pools = []
    try:
        with open(spec_d.log, errors="replace") as _f:
            for _ln in _f:
                _m = re.search(r"HiCache host KV pool \((\d+) tokens\)", _ln)
                if _m:
                    _pools.append(int(_m.group(1)))
    except OSError:
        pass
    carrier_max_tokens = int(0.9 * min(_pools)) if _pools else 0
    state.carrier_max_tokens = carrier_max_tokens
    log(f"CARRIER BOUND: group D host KV pools (tokens) = {_pools} -> front --carrier-max-tokens {carrier_max_tokens} "
        f"(0 = not found in D's log, route disabled); prompts above it are served by ONE prefill on D")
    clips = count_marker(spec_d.log, "window clip") + count_marker(spec_d.log, "Bar1WindowRefused")
    log(f"BAR1 fit (deviation: transports open): D log 'window clip'/'Bar1WindowRefused' lines = {clips} (0 = both groups fit the aperture)")

    # 6. front
    front_argv = [
        py, "-m", "sglang.srt.weg2.front",
        "--prefill", f"http://127.0.0.1:{PORT_P}", "--decode", f"http://127.0.0.1:{PORT_D}",
        "--port", str(PORT_FRONT), "--awake", "D", "--tag", ns.tag,
        "--store-dir", store_dir,
        "--prefill-sid", str(spec_p.pid), "--decode-sid", str(spec_d.pid),
        "--dc-reserve", ",".join(f"{c.uuid}={dc_expect_d[c.uuid]}" for c in cards),
        "--fairness-w-s", str(ns.fairness_w_s),
        "--weight-chunks", str(chunk_count),
        "--carrier-max-tokens", str(carrier_max_tokens),
    ]
    fenv = dict(os.environ)
    fenv["PYTHONPATH"] = f"{tree}/python"
    log("front argv: " + " ".join(shlex.quote(a) for a in front_argv))
    ffh = open(front_log, "ab")
    fp = subprocess.Popen(front_argv, env=fenv, stdout=ffh, stderr=subprocess.STDOUT, cwd=tree, start_new_session=True)
    state.pids["front"] = fp.pid
    state.t_ready["front"] = wait_ready(PORT_FRONT, fp.pid, 120, log, "front", fp)

    # 7. deadmen + helpers
    huge = 10**7
    state.helper_pids.append(arm_deadman(log, spec_p.log, PORT_P, f"launch_server.*--port {PORT_P}", huge, ns.tag, "P", dry))
    state.helper_pids.append(arm_deadman(log, spec_d.log, PORT_D, f"launch_server.*--port {PORT_D}", huge, ns.tag, "D", dry))
    state.helper_pids.append(arm_deadman(log, front_log, PORT_FRONT, "sglang.srt.weg2.front", 120, ns.tag, "front", dry))
    os.system(f"ln -sfn {shlex.quote(front_log)} /root/current_boot.log")
    with open(f"{GPU_ARB}/weg2/boot_{ns.tag}.logpath", "w") as f:
        f.write(front_log + "\n")
    _write_state(state)
    log(f"LAUNCHED: P pid {spec_p.pid} (asleep) D pid {spec_d.pid} (awake) front pid {fp.pid}; state {state_path(state)}; /root/current_boot.log -> {front_log}")
    return 0


def state_path(state: BootState) -> str:
    return f"{GPU_ARB}/weg2/boot_{state.tag}.json"


def _write_state(state: BootState) -> None:
    os.makedirs(f"{GPU_ARB}/weg2", exist_ok=True)
    with open(state_path(state), "w") as f:
        json.dump(state.__dict__, f, indent=1, default=str)


def teardown(path: str) -> int:
    st = json.load(open(path))
    print(f"[{_now()}] WEG2-TEARDOWN {path}")
    pids = set()
    for name, pid in st.get("pids", {}).items():
        if pid:
            pids |= session_pids(int(pid))
            pids.add(int(pid))
    out = subprocess.run(["pgrep", "-f", r"launch_server.*--port 3003[12]|sglang\.srt\.weg2\.front"], capture_output=True, text=True).stdout.split()
    pids |= {int(p) for p in out}
    for hp in st.get("helper_pids", []):
        if hp:
            try:
                os.kill(int(hp), signal.SIGTERM)
            except OSError:
                pass
    print(f"TERM {sorted(pids)}")
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < 60 and any(_alive(p) for p in pids):
        time.sleep(2)
    left = [p for p in pids if _alive(p)]
    for p in left:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    print(f"KILL leftovers {left}")
    time.sleep(3)
    mount = st.get("store_mount", STORE_MOUNT)
    if f" {mount} " in open("/proc/mounts").read():
        subprocess.run(["umount", mount], check=False)
        print(f"store tmpfs {mount} unmounted")
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Weg2LaunchRefused as e:
        print(f"[{_now()}] WEG2-LAUNCH REFUSED: {e}", flush=True)
        raise SystemExit(2)
    except host_ledger.Weg2HostLedgerRefused as e:
        print(f"[{_now()}] WEG2-LAUNCH REFUSED: {e}", flush=True)
        raise SystemExit(2)
