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
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import host_ledger, ring_table

MIB = 1024 * 1024
PORT_FRONT = 30030
PORT_P = 30031
PORT_D = 30032
EVIDENCE_DIR = "/spinning/evidence-665-f1"
GPU_ARB = "/spinning/gpu-arb"
#: The step-0 metal probe's RECORD (C0, WEG2_BUILD_DECISIONS_0906 section 1p).
#: A FILE, not a number: the per-card duplex ratios C12/C13 gate on are parsed
#: out of its own measured rows and printed with this path beside them.
DUPLEX_PROBE_DEFAULT = f"{GPU_ARB}/weg2/PROBE_RING_0907.md"
DEADMAN = f"{GPU_ARB}/devtools/boot_deadman.sh"
MEMTS = f"{GPU_ARB}/devtools/mem_timeseries.sh"
HOST_PREFLIGHT = f"{GPU_ARB}/devtools/host_ledger_preflight.sh"
PRESENCE_DIR = "/dev/shm/sglang-phase-flip-presence"
STORE_MOUNT = "/spinning/hicache-weg2-ram"
#: C18: where the per-card host-ring files live under the MAP_SHARED form.  A
#: tmpfs, because the granules must be shared PAGES (both co-located rank
#: processes map the same file), not a disk-backed file.
HOST_RING_DIR = "/dev/shm/weg2-hostring"
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
#: #1233 draft KV across the flip (spec section 6, Q17): group P's token
#: capacity is EXPLICIT, never left to the profiler, because the last stage
#: now carries the MTP head (405.2 MiB) plus a resident INT8 embed_tokens
#: (1213.0 MiB) = 1618.2 MiB RESIDENT, and a draft device pool at
#: target-token parity (2048 B/token beside the 8192 B/token target cell).
#: Measured weg2zr2 on the last stage: KV pool 5584.4 MiB (714,788 tokens),
#: NVML free minimum 1450 MiB. Target: idle NVML free at the corridor TOP
#: (1229 MiB) so the draft-extend transient lands inside 819-1229:
#:   pool_new = 1450 + 5584.4 - 1618.2 - 1229 = 4187 MiB
#:   T_P      = 4187 x 2^20 / (8192 + 2048) = 428,769 -> 428,000 (uniform over
#:              the stages; demand bound 8 x 27,466 = 219,728, 1.95x slack).
#: BOOT weg2dk2 (35bdc9e310, 2026-09-07 18:22:15, PP2 log): the MTP head's
#: "Load weight end" delta was 3.90 GB = 3994 MiB -- 2376 MiB more than the
#: resident post. That figure is the BUILD, not the residue: it holds the
#: head's OWN never-loaded lm_head, a bf16 [248320 x 5120] table (2426 MiB;
#: "lm_head" is in the checkpoint's quantization ignore list, so it is not
#: int8), which `Qwen3_5ForCausalLMMTP.__init__` materialises on the scope's
#: single-rank is_last_rank and which the producer drops when it shares the
#: target's head (405.2 + 1213.0 + 2426.0 = 4044, 50 MiB from the reading).
#: No second embedding is budgeted: `load_resident_embedding` loads INTO the
#: built int8 tensors. The W11 gate below refuses the boot when the measured
#: residue exceeds this budget by more than the tolerance, so the corridor
#: claim of this derivation is checked at readiness, never assumed.
#: BOOT weg2dk3 (bc31554f90, PP2 log :260) measured `resident_mib=3998.0`
#: against this line and W11 refused. TWO defects, both fixed in fix 3, and
#: the budget stands unchanged:
#:  1. fix 2 released the table by swapping the MODULE and calling
#:     empty_cache(); the upstream release form is `del lm_head.weight`
#:     (qwen3_5_mtp.py:185-193), because a module swap frees nothing while
#:     any other holder remains. Fix 3 deletes the parameters.
#:  2. the instrument could not measure the residue at all: with
#:     --enable-memory-saver the weights are loaded inside
#:     `memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS)`, which is
#:     `torch.cuda.use_mem_pool(...)` (torch_memory_saver/entrypoint.py:89-91),
#:     and `empty_cache()` does not return a MemPool block to the driver
#:     (maps/memsaver.md N4). The NVML free delta therefore reads the BUILD
#:     under this boot form no matter what is released. It is not a loss --
#:     the KV pools allocate from the SAME primary pool and reuse the block --
#:     so `resident_mib` is now the draft model's live weight bytes (shared
#:     target lm_head excluded) and the NVML delta rides beside it as its own
#:     named term `nvml_delta_mib`.
#: The next boot's L2 `resident_mib` replaces 1618.2 here, with its tag.
P_DRAFT_RESIDENT_BUDGET_MIB = 405.2 + 1213.0
P_DRAFT_RESIDENT_TOL_MIB = 256.0
P_WEG2ZR2_LAST_STAGE_NVML_FREE_MIN_MIB = 1450.0
P_WEG2ZR2_LAST_STAGE_KV_POOL_MIB = 5584.4
P_CORRIDOR_TOP_MIB = 1229.0
P_BYTES_PER_TOKEN = 8192 + 2048


def derive_p_max_total_tokens() -> int:
    pool_new = (P_WEG2ZR2_LAST_STAGE_NVML_FREE_MIN_MIB + P_WEG2ZR2_LAST_STAGE_KV_POOL_MIB
                - P_DRAFT_RESIDENT_BUDGET_MIB - P_CORRIDOR_TOP_MIB)
    return int(pool_new * 2**20 / P_BYTES_PER_TOKEN) // 1000 * 1000


P_MAX_TOTAL_TOKENS = derive_p_max_total_tokens()
assert P_MAX_TOTAL_TOKENS == 428000, P_MAX_TOTAL_TOKENS
VENV_DEFAULT = "/spinning/htsglang-gpu/.venv"


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
    ring_lines: List[str] = field(default_factory=list)
    ring_form: str = ""
    ring_dir: str = ""
    argv: Dict[str, str] = field(default_factory=dict)
    t_ready: Dict[str, float] = field(default_factory=dict)
    sleep_p_ms: float = 0.0
    deviations: List[str] = field(default_factory=list)
    carrier_max_tokens: int = 0
    weight_chunks: int = 0
    tms_so: str = ""


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


def common_flags(model: str, s_gb: int, m_mib: int, store_gib: float) -> List[str]:
    return [
        "--model-path", model,
        "--trust-remote-code",
        "--served-model-name", "Qwen3.8-27B",
        "--rank-gpu-id", "0,1,2",
        "--skip-server-warmup",
        "--disable-overlap-schedule",
        "--kv-cache-dtype", "fp8_e4m3",
        "--context-length", "262144",
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
        "--hicache-write-policy", "write_through",
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


def argv_p(py: str, model: str, budgets: List[int], s_gb: int, m_mib: int, store_gib: float, extra: List[str]) -> List[str]:
    return [py, "-m", "sglang.launch_server"] + common_flags(model, s_gb, m_mib, store_gib) + [
        "--tp-size", "1", "--pp-size", "3",
        "--pp-stage-ratio", "32,18,14", "--pp-attn-stage-ratio", "8,4,4",
        "--rank-gpu-memory-mib", ",".join(str(b) for b in budgets),
        "--barlink-bar1-window-mib", "24,PP_0=96",
        # #1233 draft KV across the flip (C15): P carries D's four speculative
        # flags BYTE-FOR-BYTE (they hash into the drafter identity, W5) and is
        # silenced by the producer flag, which is deliberately not hashed.
        "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "2",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "3",
        "--speculative-draft-kv-only",
        "--max-total-tokens", str(P_MAX_TOTAL_TOKENS),
        "--port", str(PORT_P),
    ] + extra


def argv_d(py: str, model: str, budgets: List[int], s_gb: int, m_mib: int, store_gib: float, extra: List[str]) -> List[str]:
    return [py, "-m", "sglang.launch_server"] + common_flags(model, s_gb, m_mib, store_gib) + [
        "--tp-size", "3", "--pp-size", "1",
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


def _import_duplex_gate() -> float:
    from sglang.srt.managers import weg2_memory_saver

    return float(weg2_memory_saver.DUPLEX_SPLIT_MIN_RATIO)


@dataclass
class HostRingPlan:
    """What C18 publishes, and why it may publish nothing.

    ``armed`` False means the ring is NOT in this boot, and THERE IS NO
    FALLBACK -- spec Amendment A1-3, measured, not argued:

        the OLD flip form is INFEASIBLE on this host budget on the default
        table (W20 at every rung: host weights 38.77 GiB run / 33.12 GiB
        launch, boot weg2rg1) -- no Weg-2 boot may pin --ring-table-boot
        weg2zr2 to get past it; the ring with gathered legs is the only route.

    So every un-armed arm below REFUSES BY NAME and the launcher exits 2.  The
    predecessor of this text said "the host ring is NOT armed and the OLD flip
    form runs (spec section 10.4)" in the W32, W33 and W34 arms, and the metal
    had refuted it twenty minutes before that commit was written: those lines
    are printed at ``prepare_host_ring`` (launcher.py step 1b) and the arm that
    decides whether the un-armed form is fundable is ``host_ledger.choose`` one
    step LATER, so the sentence was an assertion made before its own check --
    and the check said no.  A refusal that names a fallback must HAVE one.

    Two reasons remain, and both are printed:

    * ``table`` is not None -- W33 (no proven registration form), W32, or W34
      (the launcher found itself without gathered legs).  The table exists, so
      the un-armed charge CAN be priced; it just cannot be funded.
    * ``table`` is None -- R22, no measured table at all.  Then it cannot even
      be priced, and the ledger refuses by name (W20) for that reason instead.
    """

    form: str = ""
    dir: str = ""
    env_map: str = ""
    epoch: int = 0
    armed: bool = False
    table: Optional["ring_table.RingTable"] = None
    lines: List[str] = field(default_factory=list)
    #: C12/C13: the per-card duplex table and the ``SGLANG_WEG2_PCIE_DUPLEX``
    #: string built from it.  Launcher OUTPUT (spec R19), solved from the
    #: step-0 probe record; empty means no card splits its lock key.
    duplex: Optional["ring_table.DuplexTable"] = None
    duplex_env: str = ""

    @property
    def host_weights_bytes(self) -> int:
        """The RUN-moment host weights charge, from the same measured table for
        both forms.

        Armed: ``Sigma_c H(c)``.  The region is fungible, so the tag in flight
        is drawn from the SAME per-card bytes the dormant image occupies and
        there is nothing to add (spec C19 / section 7).

        Un-armed (the OLD serial per-tag form): one image plus ONE tag in
        flight -- ``Sigma_c H(c) + max_k Sigma_c bytes(k, c)``.  This is the
        parent's ``backup_resident + chunk_gib`` model with the stale constant
        replaced by the measured table and the AVERAGE chunk replaced by the
        measured largest one.  It is ONE tag because the front sends one
        ``/release_memory_occupation`` per tag; the sum is over cards because
        that tag's shards land on every card at once.  Charging ``Sigma_c
        max_k`` instead -- each card's own largest tag, summed -- prices a step
        the flip never takes (the maxima are different tags) and cost the
        DEFAULT path 5.6 GiB it does not spend, which was enough to make every
        arm of the ladder refuse with W20.
        """
        if self.table is None:
            return 0
        total = self.table.total_h_bytes
        if not self.armed:
            total += self.table.max_step_total_mib * ring_table.MIB
        return total

    @property
    def host_weights_span1_bytes(self) -> int:
        return 0 if self.table is None else self.table.total_span1_bytes

    @property
    def provenance(self) -> str:
        if self.table is None:
            return "no measured table"
        form = f"ring form {self.form}" if self.armed else "OLD flip form (no ring)"
        return f"{self.table.provenance()}; charged for the {form}"


#: R17's gate, read from the module that owns the key so the launcher's printed
#: verdict and the rank's actual key can never be computed from two numbers.
DUPLEX_GATE = _import_duplex_gate()


def _front_leg_form() -> str:
    """``front.FLIP_LEG_FORM`` -- read, never restated.

    The launch check has to know how the flip orders its legs, and the only
    honest source is the module that does the ordering.  A copy here would be a
    second bookkeeping of the same fact and would go stale the moment C9 lands.
    """
    from sglang.srt.weg2 import front

    return front.FLIP_LEG_FORM


def _split_decisions(
    card_uuids: Sequence[str],
    ratios: Optional[Dict[str, float]] = None,
    leg_form: str = "interleave",
    override: Optional[bool] = None,
) -> Dict[str, bool]:
    """Per card: does its PCIe key SPLIT the two directions?  Decided ONCE, here.

    FIX 1 round 1, and boot weg2rg2 is the reason it is a decision at all rather
    than a reading of R17's ratio.  R17 asks whether the split BUYS enough
    (concurrent aggregate >= 1.5x serial); weg2rg2 asked whether the single key
    is SAFE under C9's gathered legs, and the answer is no.  Each leg holds the
    per-card key across its whole tag loop, so on a one-key card the pair is
    serialised again and R5's premise -- W's per-tag releases fund S's acquires
    -- is false there: the sleep leg blocks in ``acquire`` at free = 0 holding
    the key the wake leg needs.  Measured cost of that on nvml0: 120.189 s, W31,
    W29 on three ranks, group-fatal W4.

    So: under GATHERED legs every card splits, whatever its ratio.  That is not
    free-lunch reasoning -- the x4 card's 1.316 is concurrent OVER serial, so
    concurrency is 32 % faster there too; R17's gate only ever said the split is
    not worth a mechanism of its own, and A1-4 says that card gets no BENEFIT
    from the split, not that it must not have one.  Under a front that does NOT
    gather (no state of this tree), the ratio gate stands and a card below it
    keeps one key -- which is correct there, because a serial front never has
    two legs in flight to deadlock.

    ``override`` is the test injection point: ``True`` = force every key to
    split, ``False`` = force every key single (the weg2rg2 state).
    """
    if override is not None:
        return {uuid: bool(override) for uuid in card_uuids}
    if leg_form == "interleave":
        return {uuid: True for uuid in card_uuids}
    table = ratios or {}
    return {
        uuid: table.get(uuid) is not None and table[uuid] >= DUPLEX_GATE
        for uuid in card_uuids
    }


def _serialised_cards(
    card_uuids: Sequence[str],
    published: str = "",
    leg_form: str = "interleave",
) -> List[str]:
    """The cards on which the two flip legs are SERIALISED, by UUID.

    FIX 1 (round 1), and boot weg2rg2 is why.  Serialisation is a property of a
    CARD, not of the flip: C9 puts both legs in flight, but each leg takes the
    per-card PCIe key for its WHOLE tag loop (weight_updater sleep-D2H at the
    ``with self._weg2_pcie_lock(...)`` around the pause loop, wake-H2D around
    the resume loop), and on a card whose measured duplex ratio does not reach
    R17's gate both directions resolve to the SAME key.  There the gathered pair
    is serialised again -- W's per-tag releases cannot fund S's acquires,
    because W is queued behind S on the key, or S behind W.

    The predecessor computed the AND over cards, printed it, and gated nothing;
    its justification was that "a card that keeps one key costs time on that
    card, and nothing else".  The metal refuted that 20 minutes later: nvml0
    (1.316, DUPLEX-NULL) passed R5 on both directions, armed, and wedged
    120.189 s into the FIRST flip -- W31 Weg2HostRingExhausted need=24 free=12
    MiB, its peer holding the granules and itself queued on that same single
    key, ending in W29 on all three P ranks and a group-fatal W4.

    IT ASKS THE KEY, NOT ITS OWN DICT (FIX 2 round 2, finding 2).  The
    predecessor passed :func:`_split_decisions`' answer straight back in as
    ``splits=``, so the saver returned it verbatim and the check was
    tautological -- it could not see a rank that would resolve the published
    string differently, which is the ONE thing it exists to see.  Here the
    argument is the PUBLISHED STRING, it is resolved the way a rank resolves it
    (through :data:`weg2_memory_saver.PCIE_DUPLEX_ENV`), and the answer is read
    off the KEY ITSELF: the two directions' lock paths, compared.  Any saver on
    ``PYTHONPATH`` answers that -- including one older than the wire format,
    which returns ONE path for both directions and is therefore reported here as
    SERIALISED, refused by W34 before either group starts, instead of taking one
    key on all three cards at the first flip.
    """
    if leg_form != "interleave":
        # The front itself does not gather: every card is serialised, whatever
        # its key does.
        return list(card_uuids)
    from sglang.srt.managers import weg2_memory_saver

    serialised: List[str] = []
    saved = os.environ.get(weg2_memory_saver.PCIE_DUPLEX_ENV)
    os.environ[weg2_memory_saver.PCIE_DUPLEX_ENV] = published
    try:
        for uuid in card_uuids:
            try:
                keys = {
                    weg2_memory_saver.pcie_lock_path(uuid, direction=d)
                    for d in weg2_memory_saver.PCIE_DIRECTIONS
                }
            except TypeError:
                # A saver whose pcie_lock_path has no direction parameter at
                # all: one key, both legs, by construction.
                keys = {weg2_memory_saver.pcie_lock_path(uuid)}
            except getattr(weg2_memory_saver, "Weg2DuplexDecisionRefused", ()):
                # W36 raised while resolving.  It is already a named refusal;
                # here it means SERIALISED, so the launcher's own W34 arm gives
                # the operator the whole picture in one refusal instead of a
                # traceback out of a helper.
                keys = {uuid}
            if len(keys) < len(weg2_memory_saver.PCIE_DIRECTIONS):
                serialised.append(uuid)
    finally:
        if saved is None:
            os.environ.pop(weg2_memory_saver.PCIE_DUPLEX_ENV, None)
        else:
            os.environ[weg2_memory_saver.PCIE_DUPLEX_ENV] = saved
    return serialised


def remove_vram_credit_counters(credit_dir: str = "") -> List[str]:
    """Unlink this box's per-card VRAM credit counters.  Returns what went.

    FIX 2 round 2, finding 1.  ``vram_credit_path`` is keyed by card UUID and
    nothing else, so the file outlives the boot that wrote it -- and a TERMINAL
    one (``leg_complete`` plus a whole image of credit) is exactly what a later
    boot's flip of the same index read as its own funding while the epoch was
    only the front's flip counter.  Measured on this rig 2026-09-08: three such
    files from boot weg2rg2, one carrying 13912 MiB of credit.
    """
    from sglang.srt.managers import weg2_memory_saver

    directory = credit_dir or os.environ.get(
        weg2_memory_saver.VRAM_CREDIT_DIR_ENV,
        os.environ.get(weg2_memory_saver.PCIE_LOCK_DIR_ENV,
                       weg2_memory_saver.DEFAULT_PCIE_LOCK_DIR),
    )
    prefix = "." + weg2_memory_saver.VRAM_CREDIT_PREFIX
    removed: List[str] = []
    if not os.path.isdir(directory):
        return removed
    for name in sorted(os.listdir(directory)):
        if not name.startswith(prefix):
            continue
        try:
            os.unlink(os.path.join(directory, name))
        except OSError:
            continue
        removed.append(name)
    return removed


def prepare_host_ring(cards: List[Card], log: Log, tag: str, form: str,
                      evidence_dir: str, boot_stem: str, dry: bool,
                      leg_form: str = "", pcie_directional: Optional[bool] = None,
                      duplex_probe: str = "") -> HostRingPlan:
    """C20 + C18: solve the table, print L6, REFUSE by name, then arm the region.

    Order is load-bearing: the inequalities are checked and the per-card files
    are created BEFORE either group starts, so a configuration that cannot walk
    the corridor never reaches a rank (W32/W34 rather than a mid-flip W31).

    TWO inequalities, and WHICH ONE A CARD IS CHECKED AGAINST IS THE CARD'S OWN
    PROPERTY (FIX 1 round 1).  R5's corridor is checked on every card (W32).
    The SERIAL form's ``H(c) >= image_W(c) + max_tag_S(c)`` is checked on the
    cards where the two legs cannot actually overlap (W34) -- see
    :func:`_serialised_cards`, which reads the leg form from the front and the
    key from the module that owns it, per card.  ``pcie_directional`` is the
    test override for that second fact: ``True`` = every card's key splits,
    ``False`` = none does, ``None`` = ask per card.

    NO TABLE IS NOT A FALLBACK: with ``table`` None this returns an un-armed,
    unpriceable plan and the ledger refuses the launch (W20).  See
    :class:`HostRingPlan`.
    """
    leg_form = leg_form or _front_leg_form()
    # C12/C13 + A1-4: the per-card duplex ratio, SOLVED from the step-0 probe's
    # own lines and published to the ranks as launcher output.  Printed per card
    # BEFORE either group starts, beside H(c), because the split's benefit is
    # per card and the flip's critical path is the card that does not get it.
    duplex, duplex_why = ring_table.solve_duplex(duplex_probe) if duplex_probe else (None, "no --duplex-probe given")
    plan = HostRingPlan(form=form, duplex=duplex)
    if duplex is None:
        plan.lines.append(
            f"WEG2-PCIE-DUPLEX UNMEASURED: {duplex_why} -- no card has a RATIO "
            "this boot, so none of them has an expectation about what the split "
            "buys.  Under gathered legs every key still SPLITS, and that is a "
            "correctness decision, not an optimisation: with one key the sleep "
            "leg blocks in the ring's acquire holding the key the wake leg needs "
            "(boot weg2rg2, W31 after 120.189 s -> W29 -> W4).  The predecessor "
            "of this line said an unmeasured card is 'slower, never wrong'; the "
            "metal refuted that."
        )
    else:
        plan.lines.extend(duplex.format_lines(cards, DUPLEX_GATE))
    for ln in plan.lines:
        log(ln)
    logged = len(plan.lines)
    table, reason = ring_table.solve(cards, evidence_dir, boot_stem or None)
    if table is None:
        plan.lines.append(
            "WEG2-HOST-RING R22: no measured per-card byte table -- " + reason +
            ".  The planner REFUSES to guess H, and THIS BOOT REFUSES: the OLD "
            "flip form is not a fallback here, because its own host charge (one "
            "image plus one tag in flight) is solved from the same table, so the "
            "ledger has no host weights term and stops the launch by name (W20 "
            "Weg2HostLedgerRefused, exit 2) before either group starts.  What "
            "makes a boot priceable again is a predecessor boot in "
            + (evidence_dir or "the evidence dir") +
            " that logged WEG2-CHUNK-BYTES / WEG2-FLIP-TAG lines and its own "
            "NVML -> CUDA ordinal map, or --ring-table-boot naming one."
        )
        for ln in plan.lines[logged:]:
            log(ln)
        return plan
    plan.table = table
    plan.lines.extend(table.format_l6())          # L6
    for ln in plan.lines[logged:]:
        log(ln)
    if form not in ("auto", "MAP_SHARED"):
        head = (
            f"W33 Weg2RingFormUnproven: --ring-form {form or 'none'} selects no "
            "registration form, so the host ring cannot be armed -- AND THERE IS NO "
            "FALLBACK TO RUN INSTEAD (spec Amendment A1-3: the OLD flip form is "
            "INFEASIBLE on this host budget, W20 at every rung on the default "
            "table, boot weg2rg1).  This boot REFUSES, exit 2, before either group "
            "starts.  MAP_SHARED (cudaHostRegister on a /dev/shm MAP_SHARED file) "
            "is the form the step-0 probe PROVED on this rig "
            "(WEG2_BUILD_DECISIONS_0906 section 1p, 2026-09-07T23:13:23Z: all three "
            "gates pass, verdict BUILD-MAP_SHARED, duplex arm KEEP); it is the only "
            "form built, and the memfd candidate that probe retired is deleted."
        )
        plan.lines.append(head)
        log(head)
        raise Weg2RingFormUnproven(head)
    # TWO launch checks, ONE rule for what a failure does: the ring cannot be
    # armed, and per Amendment A1-3 there is nothing to fall back to, so the
    # launcher REFUSES BY NAME and exits 2 -- whether or not a form was asked
    # for explicitly.  The predecessor downgraded silently to "the OLD flip form
    # runs (spec section 10.4)" unless --ring-form named a form; boot weg2rg1
    # then died of W20 two steps later, having told the operator a fallback had
    # run that never did.
    #
    # W32 is spec R5's corridor inequality -- the one that makes a blocking ring
    # deadlock-free when the legs are GATHERED (spec C9).  It is checked always.
    checks = [
        ("W32 Weg2RingCreditRefused",
         "the R5 corridor inequality (H(c) >= image_W(c) - device_credit(c) + "
         "max_tag_S(c) + max_tag_W(c), spec C20) fails on {n} (card x direction) "
         "case(s); a negative slack is a flip that would wedge, not one that "
         "would be slow",
         table.refusals(),
         ring_table.Weg2RingCreditRefused),
    ]
    # W34 IS PER CARD (FIX 1 round 1).  The serial requirement
    # H(c) >= image_W(c) + max_tag_S(c) applies to a card wherever the two legs
    # cannot actually overlap ON THAT CARD, and there are two ways for that to
    # be true:
    #
    #   * the FRONT does not gather at all (``FLIP_LEG_FORM != "interleave"``,
    #     which no state of this tree produces -- a stale worktree on
    #     PYTHONPATH, a partial rebase).  Then every card is serialised.
    #   * the CARD's key does not split (A1-4: 5090 1.759 and nvml2 1.687 reach
    #     R17's gate, nvml0 measured 1.316 and does not).  Each leg holds that
    #     key across its whole tag loop, so on that card the gathered pair is
    #     serialised again and R5's premise -- W's per-tag releases fund S's
    #     acquires -- is false there.
    #
    # The predecessor checked only the first and printed the second, and boot
    # weg2rg2 is what that cost: nvml0 passed R5 on both directions, the L6 line
    # said SERIAL FORM slack=1122/-2986 MiB on the SAME card, the launcher armed
    # anyway, and the first flip wedged for 120.189 s into W31 -> W29 on all
    # three P ranks -> group-fatal W4.  A refusal 120 s into a flip that has
    # already mutated VRAM is not the refusal this gate exists to give.
    card_uuids = [c.uuid for c in table.cards]
    splits = _split_decisions(
        card_uuids, duplex.ratios if duplex else {}, leg_form, pcie_directional
    )
    # THE DECISION IS PUBLISHED HERE AND NOWHERE ELSE (R19: launcher output).
    # The ranks build their key from this same string, so the inequality this
    # module checks and the key the metal takes cannot be two different facts.
    # THE FORMAT NAMES ITS VERSION (FIX 2 round 2).  A reader that does not know
    # ``v2`` refuses by name (W36) instead of dropping every row and resolving a
    # single key on every card, which is what the un-versioned predecessor did
    # to any older saver on PYTHONPATH -- strictly worse than boot weg2rg2, and
    # silent.
    from sglang.srt.managers import weg2_memory_saver as _saver
    plan.duplex_env = f"{_saver.PCIE_DUPLEX_FORMAT}|" + ",".join(
        f"{u}={('%.3f' % duplex.ratios[u]) if duplex and u in duplex.ratios else ''}"
        f":{'split' if splits[u] else 'single'}"
        for u in card_uuids
    )
    serialised = _serialised_cards(card_uuids, plan.duplex_env, leg_form)
    checks_logged = len(plan.lines)
    for c in table.cards:
        is_serial = c.uuid in serialised
        plan.lines.append(
            f"WEG2-HOST-RING CHECK card={c.uuid} nvml{c.nvml_index} {c.name} "
            f"key={'SINGLE (both legs serialise here)' if is_serial else 'SPLIT per direction'} "
            f"duplex_ratio={(duplex.ratios.get(c.uuid) if duplex else None)} "
            f"gate={DUPLEX_GATE} leg_form={leg_form} -- checked against "
            + ("the R5 corridor (W32) AND the SERIAL requirement "
               "H >= image_W + max_tag_S (W34), because the two legs cannot "
               "overlap on this card"
               if is_serial else
               "the R5 corridor (W32) alone, because both legs can be in flight "
               "on this card at once")
        )
    for ln in plan.lines[checks_logged:]:
        log(ln)
    if serialised:
        why_serial = (
            f"front.FLIP_LEG_FORM is {leg_form!r}, not 'interleave' -- this "
            "launcher is running against a front that does not gather its legs, "
            "which no state of this tree produces (C9), so EVERY card is "
            "serialised"
            if leg_form != "interleave" else
            f"the legs are gathered (C9) but on {len(serialised)} of "
            f"{len(table.cards)} card(s) -- {', '.join(serialised)} -- the "
            "measured duplex ratio does not reach R17's gate, so a sleep-D2H and "
            "the co-located wake-H2D take the SAME per-card PCIe key and each "
            "holds it across its whole tag loop; on those cards the pair is "
            "serialised whatever the front does"
        )
        checks.append(
            ("W34 Weg2RingNeedsInterleave",
             why_serial + "; the serial requirement H(c) >= image_W(c) + "
             "max_tag_S(c) therefore applies THERE and fails on {n} "
             "(card x direction) case(s).  S blocks in acquire holding the "
             "per-card key, W cannot issue the release that would fund it "
             "because it is queued on that same key, and the acquire budget "
             "expires into W31 -> W29 on every rank of the group -> "
             "group-fatal W4.  That is boot weg2rg2, observed, not predicted",
             table.serial_refusals(only=serialised),
             ring_table.Weg2RingNeedsInterleave))
    for name, why, bad, exc in checks:
        if not bad:
            continue
        head = (f"{name}: " + why.format(n=len(bad)) +
                ".  The host ring is NOT armed, and per spec Amendment A1-3 "
                "there is NO fallback flip form to run on this host budget "
                "(the OLD form refuses W20 at every rung on the default table, "
                "boot weg2rg1) -- so this boot REFUSES by name and exits 2, "
                "BEFORE either group starts.")
        plan.lines.append(head)
        plan.lines.extend(bad)
        for ln in plan.lines[-(len(bad) + 1):]:
            log(ln)
        raise exc(head + "\n" + "\n".join(bad))
    plan.epoch = int(time.time())
    plan.dir = f"{HOST_RING_DIR}-{tag}"
    if not dry:
        os.makedirs(plan.dir, exist_ok=True)
    for c in table.cards:
        path = os.path.join(plan.dir, f"{c.uuid}.ring")
        size = ring_table.MIB * 2 + c.h_mib * ring_table.MIB  # header granule + data
        if not dry:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        plan.lines.append(
            f"WEG2-HOST-RING file {path} size={size // ring_table.MIB} MiB "
            f"(2 MiB header granule + H {c.h_mib} MiB) "
            + ("DRY-RUN: would be" if dry else "")
            + " created and ftruncated BEFORE either group starts"
        )
    plan.env_map = table.env_map()
    plan.form = "MAP_SHARED"          # 'auto' resolves to the one proven form
    plan.armed = True
    plan.lines.append(
        f"WEG2-HOST-RING ARMED form={plan.form} leg_form={leg_form} "
        f"serialised_cards={serialised or 'none'} "
        f"(each checked against the SERIAL requirement, not R5's corridor -- "
        f"see the WEG2-HOST-RING CHECK line per card) "
        f"epoch={plan.epoch} dir={plan.dir} "
        f"Sigma H={table.total_h_bytes // ring_table.MIB} MiB "
        f"Sigma span1={table.total_span1_bytes // ring_table.MIB} MiB granule=2 MiB "
        f"register_est_ms={int(table.total_span1_bytes / float(2**30) * 44)} "
        "(38-49 ms/GiB measured on the step-0 probe at ONE size point, 4 GiB -- "
        "an extrapolation, and it is a LAUNCH-moment charge: span 1 is "
        "registered at P's first pause, R7) "
        f"-- provenance: {table.provenance()}"
    )
    for ln in plan.lines[-(len(table.cards) + 1):]:
        log(ln)
    return plan


def build_env(tree: str, venv: str, cvd: str, store_dir: str, debug_hold: bool, tag: str,
              chunk_layers: int = 0, chunk_count: int = 0, tms_so: str = "",
              transport: str = "bar1", ring: Optional["HostRingPlan"] = None) -> Dict[str, str]:
    env = dict(os.environ)
    # C18: the shared host granule ring (spec C1-C8).  These four variables are
    # LAUNCHER OUTPUT, never operator input (R19): every size in them is solved
    # by ring_table from the previous boot's own lines, and the whole family is
    # absent when no form is proven -- which is the ONE boot-level fallback, and
    # makes the stock cudaMallocHost path plus the OLD serial flip form run
    # (spec section 10.4 / R22).
    if ring is not None and ring.armed:
        env["TMS_HOST_RING_DIR"] = ring.dir
        env["TMS_HOST_RING_MAP"] = ring.env_map
        env["TMS_HOST_RING_EPOCH"] = str(ring.epoch)
        env["TMS_HOST_RING_FORM"] = ring.form
    else:
        for key in ("TMS_HOST_RING_DIR", "TMS_HOST_RING_MAP", "TMS_HOST_RING_EPOCH",
                    "TMS_HOST_RING_FORM"):
            env.pop(key, None)
    # C12/C13: the per-card duplex ratio, same class of variable and the same
    # rule -- launcher OUTPUT solved from a measured file, never operator input.
    # Absent means no card splits its PCIe lock key, which is the conservative
    # direction; it is popped rather than left inherited so a stale value from
    # the launcher's own environment can never open a split nobody measured.
    if ring is not None and ring.duplex_env:
        env["SGLANG_WEG2_PCIE_DUPLEX"] = ring.duplex_env
    else:
        env.pop("SGLANG_WEG2_PCIE_DUPLEX", None)
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


_DRAFTER_RE = re.compile(r"HiCache draft KV registered: .*drafter=([0-9a-f]{16}|None)")
_LAYOUT_RE = re.compile(r"#706 canonical DRAFT page active: .*layout=v(\d+) drafter=([0-9a-f]{16}|None)")


def check_drafter_identity(log_p: str, log_d: str) -> Dict[str, object]:
    """W10 (#1233): the drafter identity and canonical draft layout of both
    groups, read from their logs. ``match`` is False on ANY disagreement or
    absence -- a group that registered no drafter at all is the pre-fix
    shape, not a pass."""
    out: Dict[str, object] = {}
    for name, path in (("P", log_p), ("D", log_d)):
        ids, layouts, n = set(), set(), 0
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    m = _DRAFTER_RE.search(line)
                    if m:
                        ids.add(m.group(1))
                        n += 1
                    m = _LAYOUT_RE.search(line)
                    if m:
                        layouts.add(f"v{m.group(1)}")
        except OSError:
            pass
        out[name] = ",".join(sorted(ids)) if ids else None
        out[f"layout_{name}"] = ",".join(sorted(layouts)) if layouts else None
        out[f"n_{name}"] = n
    out["match"] = bool(
        out["P"] and out["D"] and out["P"] == out["D"] and "," not in str(out["P"])
        and out["layout_P"] and out["layout_P"] == out["layout_D"]
    )
    return out


_RESIDENT_RE = re.compile(r"WEG2 DRAFT-KV-PRODUCER armed .*resident_mib=(-?\d+(?:\.\d+)?)")


def check_draft_resident(log_p: str, budget_mib: float = None, tol_mib: float = None) -> Dict[str, object]:
    """W11 (#1233 fix 2): the producer's MEASURED resident MiB on P's last
    stage (L2 `resident_mib=`) against the budget T_P was derived from.
    ``ok`` is False on absence, on an unmeasured value (-1) and above
    budget + tolerance -- the corridor derivation is then refuted by the
    boot itself and the front must not open on it."""
    budget = P_DRAFT_RESIDENT_BUDGET_MIB if budget_mib is None else float(budget_mib)
    tol = P_DRAFT_RESIDENT_TOL_MIB if tol_mib is None else float(tol_mib)
    out: Dict[str, object] = {"resident_mib": None, "budget_mib": budget, "tol_mib": tol, "over_mib": None, "ok": False}
    try:
        with open(log_p, errors="replace") as f:
            for line in f:
                m = _RESIDENT_RE.search(line)
                if m:
                    out["resident_mib"] = float(m.group(1))
    except OSError:
        pass
    r = out["resident_mib"]
    if r is not None:
        out["over_mib"] = r - budget
        out["ok"] = r >= 0 and r <= budget + tol
    return out


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
    p = subprocess.Popen(spec.argv, env=spec.env, stdout=fh, stderr=subprocess.STDOUT, cwd=tree,
                         start_new_session=True)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    ap.add_argument(
        "--ring-form", choices=["auto", "none", "MAP_SHARED"], default="auto",
        help="C18/R16: the cross-process registration form for the shared host "
             "granule ring. MAP_SHARED (cudaHostRegister on a /dev/shm MAP_SHARED "
             "file) is the form the step-0 metal probe PROVED on this rig "
             "(WEG2_BUILD_DECISIONS_0906 section 1p); it is the only form built, "
             "and the memfd candidate is deleted because that probe retired it. "
             "'auto' (the default) arms it when the launch checks fund it, and "
             "otherwise prints the failing check by name (W32 / W34) and REFUSES "
             "the boot, exit 2 -- spec Amendment A1-3: the OLD flip form is "
             "infeasible on this host budget, so an un-armed ring has nothing to "
             "fall back to. 'none' never arms and therefore never launches. Never "
             "a size, never a knob: H and both spans are solved from the previous "
             "boot's own lines.",
    )
    ap.add_argument("--evidence-dir", default=EVIDENCE_DIR,
                    help="where ring_table reads the previous boot's logs from")
    ap.add_argument("--duplex-probe", default=DUPLEX_PROBE_DEFAULT,
                    help="C12/C13 + A1-4: the step-0 probe RECORD the per-card PCIe "
                         "duplex ratio is solved from. A measured FILE, never a "
                         "number: the launcher parses its 'PROBE granule ... ratio=' "
                         "rows per card UUID, prints one WEG2-PCIE-DUPLEX line per "
                         "card with the R17 gate beside it, and publishes the table "
                         "to the ranks as SGLANG_WEG2_PCIE_DUPLEX. An unreadable or "
                         "rowless file means NO card splits its lock key")
    ap.add_argument("--ring-table-boot", default="",
                    help="pin the ring table to ONE boot instead of the newest usable "
                         "one. Matched as a SUBSTRING of the log stem, so the boot TAG "
                         "('weg2zr2') is enough; it must match exactly one boot, and a "
                         "pin that matches none or several returns the R22 reason and "
                         "runs the OLD flip form rather than raising an OSError")
    ap.add_argument("--teardown", default="", help="path of a boot state json to tear down")
    ns = ap.parse_args(argv)
    if ns.teardown:
        return teardown(ns.teardown)

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
        "flip (C9, gathered legs) = src.pause(kv) -> ONE src.release(family) and ONE dst.resume(family) "
        "in flight together -> dst.resume(kv); the host holds ONE image per card (H(c) = max_g image_g(c)) "
        "and dst's per-tag releases fund src's acquires inside it")
    tms_so = "" if dry else build_tms_preload(tree, ns.venv, log)
    state.weight_chunks = chunk_count
    state.tms_so = tms_so

    # 1b. C20: the R5 corridor inequality per card per direction, and C18: the
    # per-card region, both BEFORE either group starts.  The table it is solved
    # from also carries the ledger's host weights term, so this runs first.
    ring_plan = prepare_host_ring(cards, log, ns.tag, ns.ring_form, ns.evidence_dir,
                                  ns.ring_table_boot, dry, duplex_probe=ns.duplex_probe)
    state.ring_lines = ring_plan.lines
    state.ring_form = ring_plan.form if ring_plan.armed else "none (OLD flip form)"
    state.ring_dir = ring_plan.dir if (ring_plan.armed and ring_plan.form == "MAP_SHARED") else ""

    # 2. host ledger
    mi = host_ledger.read_meminfo()
    arm, store_gib, lines = host_ledger.choose(
        mi["MemTotal"], mi["MemAvailable"], store_min_gib=ns.store_min_gib,
        ring_bytes=ring_plan.host_weights_bytes,
        ring_span1_bytes=ring_plan.host_weights_span1_bytes,
        ring_provenance=ring_plan.provenance,
    )
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
    env_p = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("P", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan)
    # #1233 zero-remainder: group P ends every prefill's last chunk at N-1 and
    # publishes the recurrent anchor there (schedule_policy END-OF-PREFILL
    # ANCHOR); D can claim at most N-1 tokens of a prompt, so this is the
    # anchor it resumes from. P only: D's finish anchors serve the NEXT turn.
    env_p["SGLANG_WEG2_END_ANCHOR"] = "1"
    spec_p = GroupSpec("P", PORT_P, transport_argv(argv_p(py, ns.model, budgets_p, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_p)), ns.transport), state.logs["P"], env_p)
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
        env_d = build_env(tree, ns.venv, cvd, store_dir, False, ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan)
        spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d)), ns.transport), state.logs["D"], env_d)
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
    env_d = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("D", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan)
    spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d)), ns.transport), state.logs["D"], env_d)
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
    # #1233 draft KV across the flip: W10 DRAFTER-IDENTITY GATE. Both groups
    # register a drafter (P: the producer on its last stage, D: the NEXTN
    # worker on every rank); the identity in their `HiCache draft KV
    # registered` lines and the canonical draft layout in their `#706
    # canonical DRAFT page active` lines must agree, or D fetches
    # `{hash}.draft-<Pid>` pages P never wrote (the weg2zr2 shape, 194,088
    # failed draft fetches). Refused before the front opens.
    w10 = check_drafter_identity(spec_p.log, spec_d.log)
    log(f"W10 DRAFTER-IDENTITY P={w10['P']} D={w10['D']} layout_P={w10['layout_P']} layout_D={w10['layout_D']} "
        f"match={w10['match']} (P lines {w10['n_P']}, D lines {w10['n_D']})")
    if not w10["match"]:
        raise Weg2LaunchRefused(f"W10 Weg2DrafterIdentityMismatch: P={w10['P']} D={w10['D']} layout_P={w10['layout_P']} "
                                f"layout_D={w10['layout_D']} -- the decode group would ask the carrier for draft pages under "
                                f"an identity the prefill group never writes")
    # #1233 fix 2: W11 DRAFT-RESIDENT gate. T_P (P_MAX_TOTAL_TOKENS) is derived
    # from a budgeted residue on P's last stage; the L2 line carries the
    # MEASURED one. Over budget = the corridor derivation is refuted by this
    # boot -> refuse before the front opens (boot weg2dk2's 3994 MiB build).
    w11 = check_draft_resident(spec_p.log)
    log(f"W11 DRAFT-RESIDENT P last stage resident_mib={w11['resident_mib']} budget_mib={w11['budget_mib']:.1f} "
        f"tol_mib={w11['tol_mib']:.0f} over_mib={w11['over_mib']} ok={w11['ok']} (T_P={P_MAX_TOTAL_TOKENS})")
    if not w11["ok"]:
        raise Weg2LaunchRefused(f"W11 Weg2DraftResidentOverBudget: measured resident_mib={w11['resident_mib']} vs budget "
                                f"{w11['budget_mib']:.1f}+{w11['tol_mib']:.0f} MiB -- T_P={P_MAX_TOTAL_TOKENS} was derived for the "
                                f"budget and the last stage's corridor target (idle NVML free at {P_CORRIDOR_TOP_MIB:.0f}) cannot hold")
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
    # C14 / FIX 2 round 2: the BOOT half of the VRAM credit epoch.  Launcher
    # OUTPUT in exactly the class of TMS_HOST_RING_MAP (R19), never an operator
    # knob: it is this boot's ring epoch, the same nonce every rank already got
    # from build_env, so the front's composed <boot>.<flip> token and the ranks'
    # ring identity name ONE boot.  Absent (no armed ring) the front falls back
    # to its own start time, which is boot-unique for the same reason.
    if ring_plan is not None and ring_plan.armed:
        fenv["TMS_HOST_RING_EPOCH"] = str(ring_plan.epoch)
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
    # C18: the per-card ring files are tmpfs pages charged to this boot's host
    # ledger.  Leaving them behind would carry Sigma H of RAM into the NEXT
    # boot's baseline, where nothing names it.
    ring_dir = st.get("ring_dir", "")
    if ring_dir and os.path.isdir(ring_dir):
        for name in os.listdir(ring_dir):
            try:
                os.unlink(os.path.join(ring_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(ring_dir)
        except OSError:
            pass
        print(f"host ring dir {ring_dir} removed")
    # C14 / FIX 2 round 2, finding 1: the per-card VRAM credit counters are
    # /dev/shm pages of THIS boot, and a terminal one left behind
    # (leg_complete + a whole image of credit) is what the next boot's flip of
    # the same index used to read as its own funding.  The composed epoch makes
    # that harmless; removing the file makes it absent.  Two halves, because
    # teardown alone cannot clean up after a boot that crashed.
    print("vram credit counters removed: " + str(remove_vram_credit_counters()))
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout)
    return 0


class Weg2RingFormUnproven(ring_table.Weg2RingRefused):
    """W33: ``--ring-form`` named no form this tree has a proven build for.

    A subclass of the ring's own refusal base, so it inherits ``cli()``'s
    handler -- the one named line and exit 2 -- without being enumerated
    anywhere (FIX 2's lesson, applied to the new member rather than repeated).
    """


#: Every refusal class that must leave this launcher as the ONE named line and
#: exit 2, never as a traceback.  ``Weg2RingRefused`` is a BASE class on purpose
#: (FIX 2): the previous list enumerated its members, ``Weg2RingNeedsInterleave``
#: was not among them, and an explicitly requested ``--ring-form MAP_SHARED`` on
#: this rig therefore exited 1 with a stack trace -- which any wrapper keying on
#: the exit code reads as a crash rather than as the refusal it is.  A refusal
#: added to ring_table now inherits this handler instead of needing a line here.
REFUSALS = (Weg2LaunchRefused, ring_table.Weg2RingRefused, host_ledger.Weg2HostLedgerRefused)


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """``main`` plus the one refusal handler.  Returns the exit code.

    A function rather than a bare ``__main__`` block so the handler is
    REACHABLE FROM A TEST: the defect it fixes was an except list that had gone
    out of step with the exceptions raised, and nothing could see it.
    """
    try:
        return main(argv)
    except REFUSALS as e:
        print(f"[{_now()}] WEG2-LAUNCH REFUSED: {e}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
