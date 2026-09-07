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
  the 3080 aperture (24+96 for P, 16+32+24 for D = 192 of 224 MiB usable).
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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
DC_MEASURED_D_5090_MIB = 2228
DC_MEASURED_D_3080_MIB = 1922
DC_RESERVE_SLACK_MIB = 64
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
        "--barlink-bar1-window-mib", "16,TP_0=32,DCP_0=24",
        "--port", str(PORT_D),
    ] + extra


def build_env(tree: str, venv: str, cvd: str, store_dir: str, debug_hold: bool, tag: str,
              chunk_layers: int = 0, chunk_count: int = 0, tms_so: str = "") -> Dict[str, str]:
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
    env["SGLANG_ARMING_FLOOR_SOLVED"] = "1"
    env["SGLANG_UNEVEN_DCP"] = "1"
    env["SGLANG_UNEVEN_DCP_WEIGHTED"] = "1"
    env["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
    env["SGLANG_BARLINK_BUILD_WINDOW_CAP_S"] = env.get("SGLANG_BARLINK_BUILD_WINDOW_CAP_S", "60")
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
    dc_expect_d = {
        c.uuid: (DC_MEASURED_D_5090_MIB if "5090" in c.name else DC_MEASURED_D_3080_MIB) + DC_RESERVE_SLACK_MIB
        for c in cards
    }
    state.dc_expect_d = dc_expect_d
    log("dormant residue RESERVE for group D = MEASURED D_c(D) of boot weg2ls1b2 (2228 / 1922 / 1922 MiB, "
        f"NVML per-process, windows included) + {DC_RESERVE_SLACK_MIB} MiB slack; spec 1.6 expectation was "
        f"{DC_EXPECT_5090_MIB}/{DC_EXPECT_3080_MIB} (exceeded); graded by W19 at D's first sleep: "
        + ", ".join(f"nvml{c.nvml_index}={dc_expect_d[c.uuid]}" for c in cards))
    budgets_p = budgets_from_dc(
        cards, dc_expect_d, log, "P", overshoot_mib=P_OVERSHOOT_MIB, overshoot_provenance="boot weg2ls2b2"
    )
    state.budgets["P"] = budgets_p
    env_p = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("P", "both"), ns.tag, chunk_layers, chunk_count, tms_so)
    spec_p = GroupSpec("P", PORT_P, argv_p(py, ns.model, budgets_p, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_p)), state.logs["P"], env_p)
    state.argv["P"] = " ".join(shlex.quote(a) for a in spec_p.argv)
    state.deviations = [
        "transports stay OPEN across sleep (barlink_reopen() unwired this round; BAR1 windows sized to fit both groups: P 24+96, D 16+32+24 MiB)",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 on both groups (K2); /health_generate only via the front to the awake group",
        "group deadmen tier 1 only (PROBE_S spaced past the boot); tier 2 on the front's deadman",
        "no disk tier for the store (file backend has no second tier); tmpfs sized by the ledger",
        "the epoch is carried in the front's ledger only (no group echo/checksum on the RPC)",
        "torch_memory_saver 0.0.9.post1 preload hook rebuilt from the vendored csrc with ONE patch (cpu backup freed after resume, "
        "python/sglang/srt/weg2/tms_csrc/PATCH.md); the stock wheel is untouched and used when SGLANG_WEG2_TMS_PRELOAD_SO is unset",
        f"weights paused/resumed as {chunk_count} weights_<k> chunk tags + the base tag (#1233 one-backup flip); cuda_graph stays resident (not in the sleep tag set, as in round 1)",
    ]
    for d in state.deviations:
        log(f"DEVIATION (declared): {d}")
    launch_group(spec_p, tree, log, dry)
    if dry:
        budgets_d = budgets_from_dc(cards, {c.uuid: dc_expect_d[c.uuid] + P_WINDOWS_MIB - D_WINDOWS_MIB for c in cards}, log, "D(dry, expectation)")
        env_d = build_env(tree, ns.venv, cvd, store_dir, False, ns.tag, chunk_layers, chunk_count, tms_so)
        spec_d = GroupSpec("D", PORT_D, argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d)), state.logs["D"], env_d)
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
    env_d = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("D", "both"), ns.tag, chunk_layers, chunk_count, tms_so)
    spec_d = GroupSpec("D", PORT_D, argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d)), state.logs["D"], env_d)
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
