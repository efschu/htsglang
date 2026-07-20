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
"""Auto-performance uneven-TP split mode (``--rank-tp-ratio auto-performance``).

Builds on the measured M22 feasibility findings (see ROADMAP "Performance-
oriented uneven split"): decode throughput is FLAT (+-2%) across every
representable split on heterogeneous rigs, so the attention/GDN/KV split
STAYS the VRAM-auto split. The mode's stage-1 levers are:

  (a) the fine-grained dense-MLP unit vector (--rank-mlp-ratio family plan):
      family-selective concentration of MLP units toward compute-strong
      ranks is a measured strict prefill/throughput win (M22 C6:
      auto + 5,1,1 = +10% prefill, +7% concurrent, ~0 context) because the
      INT4 MLP bytes move while attention/KV-head and GDN/SSM splits (the
      context-expensive families) stay put;
  (b) TP-degree reduction ("drop the slow-linked card", +55-76% prefill at
      -72% context per M22) -- emitted as a RECOMMENDATION LOG only, never
      applied silently.

SSM/GDN shifting is deliberately NOT a lever: the mamba state pool moves
with the GDN units (~4.7 MiB/req/unit) and collapses context (M22 C3).

Concentration is bounded by the decode-knee guard: no rank's share of the
streamed weight bytes may exceed its share of the rig's memory bandwidth
(measured M23: beyond that knee decode regresses (-4.8% at 16,1,2) while
prefill gains saturate at the knee-exact C6 level).

The MLP vector is derived from a MEASURED hardware profile (stage-0
micro-probe: per-card GEMM + memory-bandwidth score, pairwise NCCL link
matrix), cached under ~/.cache/sglang keyed by (sorted GPU UUIDs, driver
version) so the probe runs once per rig. The chosen vector is printed as a
PINNABLE hint (same UX as SGLANG_UNEVEN_TOKEN_VECTOR): passing
``--rank-tp-ratio auto --rank-mlp-ratio <vector>`` reproduces the split with
no probe and no optimizer. SGLANG_PERF_REPROBE=1 forces a re-probe.

Context floor (``--rank-perf-loose-ctx-percent X``): every candidate's max
context is PREDICTED with the same per-rank capacity math the pool sizing
uses (budgets -> weight/mamba/reserve terms -> per-rank token capacity P_r
-> C = min_r(P_r/ratio_r) * sum(ratios), continuous optimum min(sum P,
64*min P)); only candidates with C >= (100-X)% of the VRAM-auto split's
prediction are admissible. X=0 (default) admits only free gains -- which
exist, because MLP-only shifts conserve the summed free bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROFILE_VERSION = 1
PROFILE_CACHE_DIR = os.path.expanduser("~/.cache/sglang")

#: Probe shapes (model-relevant): GEMM = one chunked-prefill MLP matmul
#: (2048 tokens x 5120 x 17408 bf16), MEMBW = a decode-style weight-streaming
#: GEMV over a ~1.3 GB bf16 matrix (the lm_head-class shape of m20).
_PROBE_GEMM_M, _PROBE_GEMM_K, _PROBE_GEMM_N = 2048, 5120, 17408
_PROBE_GEMV_ROWS, _PROBE_GEMV_K = 131072, 5120

# ---------------------------------------------------------------------------
# Capacity-prediction constants (calibrated against the M22/M23 boot logs of
# this fork's uneven-TP pipeline; see HANDOFF M22/M23).
# ---------------------------------------------------------------------------
#: Per-rank overhead (CUDA context share inside the budget, NCCL buffers,
#: attention workspaces, allocator slack) subtracted from the byte budget
#: before converting to tokens. Uniform across ranks; only candidate-relative
#: and rank-relative fidelity matters for the floor decision.
_PREDICT_OVERHEAD_MIB = 1280
#: The auto-mamba activation reserve (MAMBA_AUTO_ACTIVATION_RESERVE_MIB).
_PREDICT_MAMBA_ACT_RESERVE_MIB = 1024
#: Minimum viable per-rank token capacity: below this the weighted-DCP owner
#: rule degenerates (a rank must own >= 1 of every virtual block; M22 C3b
#: measured the resulting context collapse).
_PREDICT_MIN_RANK_TOKENS = 4096
#: Token-vector granularity of the weighted-DCP converged optimum.
_PREDICT_TOKEN_UNITS = 64
#: GEMM efficiency assumed when converting probe TFLOPS into per-token
#: prefill compute time (cancels in candidate ratios; kept for logging).
_PREDICT_GEMM_EFF = 0.6
#: Exponent of the link-bandwidth penalty folded into a rank's prefill
#: score: a compute-strong card behind a narrow link attracts fewer units.
_PREDICT_LINK_ALPHA = 0.25
#: Decode-knee guard tolerance: a candidate may not raise any rank's share
#: of the total streamed weight bytes beyond that rank's share of the rig's
#: total memory bandwidth (times 1+tol). Decode is bandwidth-bound and
#: measured FLAT below this point (M20/M22); beyond it the strong card
#: becomes the decode lockstep bottleneck AND the extra prefill gain does
#: not materialize (M23 measured: 16,1,2 at 56.5% bytes-share on a 51.9%
#: membw-share card = decode -4.8%, prefill +10.0% -- the SAME +10% the
#: knee-exact C6 vector 5,1,1 delivers at decode +0.8%). The guard is the
#: trust region of the prefill model as much as a decode protection.
_PREDICT_DECODE_KNEE_TOL = 0.02
#: MLP unit grids finer than this land close to the continuous decode knee,
#: so the strict byte-share test alone is trustworthy. Coarser grids (FP8
#: dense ~136 units, GGUF K-quant ~68 units) can only realize the split in
#: large steps, and the nearest representable vector can sit measurably ABOVE
#: the knee even when its whole-model streamed-byte share still reads under
#: the bandwidth share (M27d: FP8 4,1,1 = 51.5% bytes-share < 51.9% membw
#: share, yet decode -14/-24% vs the auto split -- the per-layer lockstep
#: bottleneck bites before the whole-model share crosses). For such coarse
#: grids we require extra headroom below the knee (see below).
_PREDICT_KNEE_COARSE_UNITS = 256
#: On a coarse MLP grid, require this many unit-steps of streamed-byte-share
#: headroom below a rank's bandwidth share before admitting a candidate, so
#: the optimizer rounds DOWN to the last safe vector instead of the lumpy
#: overshoot. Calibrated on the M27d rig (5090 + 2x3080, membw share 51.9%):
#: this rejects the FP8 4,1,1 knee overshoot (picking the 3,1,1 class) while
#: the fine AWQ 544-unit grid is unaffected and keeps its measured-win 5,1,1.
_PREDICT_KNEE_COARSE_HEADROOM_UNITS = 2
#: TP-degree recommendation: a GPU whose best pairwise link is below this
#: fraction of the rig's best link is called out as a drop candidate.
_TP_DROP_LINK_FRACTION = 0.7
#: ... provided the remaining ranks' budgets still fit the weights with
#: this fill factor of headroom.
_TP_DROP_FIT_FACTOR = 0.85


# ---------------------------------------------------------------------------
# Stage 0: hardware micro-probe (runs in a SUBPROCESS so the launcher stays
# free of CUDA state; a few seconds per rig, cached afterwards).
# ---------------------------------------------------------------------------


def _nvml_gpu_inventory() -> Tuple[List[dict], str]:
    """Per-CUDA-device {uuid, name, total_mib} (CUDA enumeration order,
    bridged to NVML via PCI bus ids like server_args does) + driver version."""
    import pynvml
    import torch

    from sglang.srt.server_args import _torch_to_nvml_gpu_index_mapping

    mapping = _torch_to_nvml_gpu_index_mapping()
    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        gpus = []
        for cuda_idx in range(torch.cuda.device_count()):
            nvml_idx = mapping.get(cuda_idx, cuda_idx)
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            total_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total // 2**20
            gpus.append(
                {
                    "cuda_index": cuda_idx,
                    "uuid": uuid,
                    "name": name,
                    "total_mib": total_mib,
                }
            )
        return gpus, driver
    finally:
        pynvml.nvmlShutdown()


def profile_cache_path(uuids: Sequence[str], driver: str) -> str:
    key = json.dumps([sorted(uuids), driver, PROFILE_VERSION])
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(PROFILE_CACHE_DIR, f"hw_profile-{digest}.json")


def _bench_gemm_tflops(dev) -> float:
    import torch

    a = torch.randn(_PROBE_GEMM_M, _PROBE_GEMM_K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(_PROBE_GEMM_K, _PROBE_GEMM_N, dtype=torch.bfloat16, device=dev)
    fn = lambda: a @ b
    for _ in range(10):
        fn()
    torch.cuda.synchronize(dev)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    iters = 60
    s.record(torch.cuda.current_stream(dev))
    for _ in range(iters):
        fn()
    e.record(torch.cuda.current_stream(dev))
    torch.cuda.synchronize(dev)
    ms = s.elapsed_time(e) / iters
    flops = 2.0 * _PROBE_GEMM_M * _PROBE_GEMM_K * _PROBE_GEMM_N
    del a, b
    torch.cuda.empty_cache()
    return flops / (ms / 1e3) / 1e12


def _bench_membw_gbs(dev) -> float:
    import torch

    x = torch.randn(1, _PROBE_GEMV_K, dtype=torch.bfloat16, device=dev)
    w = torch.randn(_PROBE_GEMV_ROWS, _PROBE_GEMV_K, dtype=torch.bfloat16, device=dev)
    fn = lambda: torch.nn.functional.linear(x, w)
    for _ in range(10):
        fn()
    torch.cuda.synchronize(dev)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    iters = 50
    s.record(torch.cuda.current_stream(dev))
    for _ in range(iters):
        fn()
    e.record(torch.cuda.current_stream(dev))
    torch.cuda.synchronize(dev)
    ms = s.elapsed_time(e) / iters
    gb = _PROBE_GEMV_ROWS * _PROBE_GEMV_K * 2 / 1e9
    del x, w
    torch.cuda.empty_cache()
    return gb / (ms / 1e3)


def _link_worker(rank: int, world: int, results) -> None:
    """NCCL pairwise link probe (adapted from m20_nccl_bench): P2P 1 MiB
    bandwidth per GPU pair + small all-reduce latency over the full group."""
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device(f"cuda:{rank}")
    out = {}

    def bench(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0  # us

    t = torch.randn(5120, dtype=torch.bfloat16, device=dev)
    out["ar_10kb_us"] = bench(lambda: dist.all_reduce(t))
    big = torch.randn(512 * 1024, dtype=torch.bfloat16, device=dev)
    out["ar_1mb_us"] = bench(lambda: dist.all_reduce(big))

    numel = 512 * 1024  # 1 MiB bf16
    for a in range(world):
        for b in range(a + 1, world):
            buf = torch.randn(numel, dtype=torch.bfloat16, device=dev)
            if rank == a:
                fn = lambda buf=buf, b=b: dist.send(buf, b)
            elif rank == b:
                fn = lambda buf=buf, a=a: dist.recv(buf, a)
            else:
                fn = lambda: None
            us = bench(fn, iters=60)
            if rank == a:
                out[f"p2p_{a}_{b}_gbs"] = numel * 2 / 1e9 / (us / 1e6)
    results[rank] = out
    dist.barrier()
    dist.destroy_process_group()


def run_probe(out_path: str) -> dict:
    """Execute the full stage-0 probe on every visible CUDA device and write
    the JSON profile to `out_path`. Meant to run inside the dedicated probe
    subprocess (see get_hardware_profile)."""
    import torch

    t0 = time.time()
    gpus, driver = _nvml_gpu_inventory()
    per_gpu: Dict[str, dict] = {}
    for g in gpus:
        dev = torch.device(f"cuda:{g['cuda_index']}")
        torch.cuda.set_device(dev)
        gemm = _bench_gemm_tflops(dev)
        membw = _bench_membw_gbs(dev)
        per_gpu[g["uuid"]] = {
            "name": g["name"],
            "cuda_index": g["cuda_index"],
            "total_mib": g["total_mib"],
            "gemm_tflops": round(gemm, 2),
            "membw_gbs": round(membw, 1),
        }

    links: Dict[str, dict] = {}
    world = torch.cuda.device_count()
    if world > 1:
        import torch.multiprocessing as mp

        mgr = mp.Manager()
        results = mgr.dict()
        mp.spawn(_link_worker, args=(world, results), nprocs=world, join=True)
        by_idx = {g["cuda_index"]: g["uuid"] for g in gpus}
        r0 = dict(results[0]) if 0 in results else {}
        for a in range(world):
            ra = dict(results[a]) if a in results else {}
            for b in range(a + 1, world):
                key = "|".join(sorted([by_idx[a], by_idx[b]]))
                gbs = ra.get(f"p2p_{a}_{b}_gbs")
                if gbs is not None:
                    links[key] = {"p2p_gbs": round(gbs, 2)}
        if "ar_10kb_us" in r0:
            links["__group__"] = {
                "ar_10kb_us": round(r0["ar_10kb_us"], 1),
                "ar_1mb_us": round(r0["ar_1mb_us"], 1),
            }

    profile = {
        "version": PROFILE_VERSION,
        "driver": driver,
        "uuids": sorted(per_gpu),
        "gpus": per_gpu,
        "links": links,
        "probe_seconds": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1)
    os.replace(tmp, out_path)
    return profile


def get_hardware_profile() -> Tuple[Optional[dict], str, List[dict]]:
    """The rig's hardware profile: from the cache when the (sorted GPU UUIDs,
    driver version) key matches, otherwise via a fresh probe subprocess
    (isolated so the launcher process stays free of CUDA contexts).

    Returns (profile or None, source description, per-CUDA-device inventory).
    SGLANG_PERF_REPROBE=1 forces a re-probe."""
    gpus, driver = _nvml_gpu_inventory()
    uuids = [g["uuid"] for g in gpus]
    path = profile_cache_path(uuids, driver)

    from sglang.srt.environ import envs

    force = bool(envs.SGLANG_PERF_REPROBE.get())
    if not force and os.path.exists(path):
        try:
            with open(path) as f:
                profile = json.load(f)
            if (
                profile.get("version") == PROFILE_VERSION
                and profile.get("driver") == driver
                and profile.get("uuids") == sorted(uuids)
            ):
                return profile, f"cache ({path})", gpus
            logger.warning(
                "auto-performance: cached profile %s has a stale key; re-probing.",
                path,
            )
        except Exception as e:
            logger.warning(
                "auto-performance: could not read cached profile %s (%s); "
                "re-probing.",
                path,
                e,
            )

    reason = "forced by SGLANG_PERF_REPROBE=1" if force else "no cached profile"
    logger.info(
        "auto-performance: running the stage-0 hardware micro-probe (%s; "
        "GEMM + memory-bandwidth per card, pairwise NCCL link matrix; "
        "a few seconds, cached to %s afterwards)...",
        reason,
        path,
    )
    cmd = [sys.executable, "-m", "sglang.srt.uneven_perf", "--probe", "--out", path]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
        if proc.returncode != 0:
            logger.warning(
                "auto-performance: hardware probe failed (rc=%d):\n%s",
                proc.returncode,
                (proc.stderr or proc.stdout or "")[-2000:],
            )
            return None, "probe failed", gpus
        with open(path) as f:
            profile = json.load(f)
        return profile, f"fresh probe ({profile.get('probe_seconds', '?')} s)", gpus
    except Exception as e:
        logger.warning("auto-performance: hardware probe failed (%s).", e)
        return None, "probe failed", gpus


def get_cached_hardware_profile() -> Tuple[Optional[dict], List[dict]]:
    """Cache-only variant of ``get_hardware_profile``: returns the cached
    profile when its (sorted GPU UUIDs, driver, version) key matches, else
    (None, inventory). NEVER triggers a probe -- used by consumers that only
    want opportunistic access to the measured scores (--rank-vocab-ratio
    auto), where a multi-second probe would be a surprising side effect."""
    gpus, driver = _nvml_gpu_inventory()
    uuids = [g["uuid"] for g in gpus]
    path = profile_cache_path(uuids, driver)
    if os.path.exists(path):
        try:
            with open(path) as f:
                profile = json.load(f)
            if (
                profile.get("version") == PROFILE_VERSION
                and profile.get("driver") == driver
                and profile.get("uuids") == sorted(uuids)
            ):
                return profile, gpus
        except Exception:
            pass
    return None, gpus


def vocab_ratio_from_membw(membw_gbs: Sequence[float], base: int = 6) -> List[int]:
    """Integer weight vector proportional to the per-rank memory-bandwidth
    scores, for ratio-weighted vocab sharding (--rank-vocab-ratio auto).

    The lm_head matvec is bandwidth-bound (it streams the whole vocab shard
    once per forward), so shard widths proportional to membw equalize the
    per-rank read TIME. Scaled so the smallest rank gets `base` units
    (granularity ~ base/sum, ample for the 64-row padded vocab units), then
    gcd-reduced. Example: 1558/723/723 GB/s -> [13, 6, 6]."""
    low = min(membw_gbs)
    assert low > 0, f"non-positive membw scores: {membw_gbs}"
    scaled = [max(1, round(b / low * base)) for b in membw_gbs]
    g = math.gcd(*scaled)
    return [s // g for s in scaled]


# ---------------------------------------------------------------------------
# Cost model: per-rank weight bytes + capacity prediction from the model
# config, mirroring the terms the real pool sizing pays (M22 cost-model
# musts: SSM pool moves with GDN units x concurrency, BF16 families inside
# INT4 checkpoints, spec-decode draft weights [embed/lm_head dupes are shared
# BEFORE profiling since eb764a12b, so only the draft's own layer shards and
# fc remain], graph/activation reserves).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlanInputs:
    """The small, boot-free input contract of the parse-time cost model.

    ``PerfCostModel`` (and the offline planner, ``sglang.srt.planner``)
    consume this dataclass instead of a full ``ServerArgs`` object, so the
    capacity math is callable as a pure library — the design-#97 "single
    source of truth" guarantee: the boot path builds a ``PlanInputs`` from
    itself (``from_server_args``) and the offline planner builds one from
    CLI/manual inputs, and both run the IDENTICAL sizing code.

    Only stdlib types; importing/constructing this never touches torch,
    CUDA, or NVML.
    """

    # -- model + speculative config (what PerfCostModel reads) --------------
    tp_size: int
    model_path: str
    kv_cache_dtype: str = "auto"
    speculative_algorithm: Optional[str] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_draft_model_path: Optional[str] = None
    max_running_requests: Optional[int] = None
    disable_cuda_graph: bool = False

    # -- placement + per-card budget (design §2.5) ---------------------------
    #: rank -> physical GPU index (duplicates = co-located ranks).
    rank_gpu_id: Optional[List[int]] = None
    #: Per-rank absolute byte budget ceiling in MiB. This is the BUDGETED
    #: value (NVML total minus the user's free-reserve minus the auto
    #: reserve), not the physical maximum — every capacity number derived
    #: from it is "max possible under your set budget".
    effective_vram_mib: Optional[List[int]] = None

    # -- manual overrides (design §2.6); None => auto-derive that knob -------
    rank_tp_ratio: Optional[List[int]] = None
    rank_mlp_ratio: Optional[List[int]] = None
    rank_moe_ratio: Optional[List[int]] = None
    rank_vocab_ratio: Optional[List[int]] = None
    dcp_size: Optional[int] = None
    kv_token_vector: Optional[List[int]] = None

    @property
    def rank_gpu_memory_mib(self):
        """Alias so functions that duck-type a ``ServerArgs`` (e.g.
        ``resolve_cp_token_ratios``) accept a ``PlanInputs`` unchanged."""
        return self.effective_vram_mib

    @classmethod
    def from_server_args(cls, server_args) -> "PlanInputs":
        """Build the cost-model inputs from a (post-validation) ServerArgs.

        Called on the boot path right before ``PerfCostModel`` is
        constructed, so server and offline planner share one input shape.
        """
        budgets = getattr(server_args, "rank_gpu_memory_mib", None)
        if isinstance(budgets, int):
            budgets = [budgets] * server_args.tp_size
        ratio = getattr(server_args, "rank_tp_ratio", None)
        return cls(
            tp_size=server_args.tp_size,
            model_path=server_args.model_path,
            kv_cache_dtype=str(server_args.kv_cache_dtype or "auto"),
            speculative_algorithm=server_args.speculative_algorithm,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            speculative_draft_model_path=server_args.speculative_draft_model_path,
            max_running_requests=server_args.max_running_requests,
            disable_cuda_graph=bool(
                getattr(server_args, "disable_cuda_graph", False)
            ),
            rank_gpu_id=(
                list(server_args.rank_gpu_id)
                if getattr(server_args, "rank_gpu_id", None)
                else None
            ),
            effective_vram_mib=list(budgets) if budgets else None,
            rank_tp_ratio=list(ratio) if isinstance(ratio, list) else None,
            dcp_size=getattr(server_args, "dcp_size", None),
        )


@dataclasses.dataclass
class _Family:
    """One weight family: total parameter count, bytes per parameter, and
    how it shards across ranks ('attn'/'gdn' follow the base plan on their
    unit grid, 'mlp' follows the candidate vector, 'even' splits evenly,
    'replicated' is per-rank constant)."""

    params: float
    bytes_per_param: float
    shard: str  # attn | gdn | mlp | even | replicated

    @property
    def bytes(self) -> float:
        return self.params * self.bytes_per_param


# ---------------------------------------------------------------------------
# GGUF metadata reader (dependency-free; mirrors the header-only reader in
# the fork's GGUF plugin). Needed because for a GGUF checkpoint the model
# path is a single .gguf FILE, so the HF ``config.json`` the cost model would
# otherwise open does not exist next to it (older builds crashed here with
# NotADirectoryError). We read the fields the cost model needs straight from
# the GGUF key/value header, and derive per-weight-family bytes/element from
# the tensor-info block's ggml quant types so the family byte model is
# roughly correct for quantized GGUF checkpoints (embed/lm_head, attention
# and MLP are quantized here, unlike the "BF16-inside-INT4" safetensors
# assumption). Duplicating ~30 lines of header parsing keeps this module
# self-contained (no import from the model loader) and the scope clean.
# ---------------------------------------------------------------------------

#: ggml quant type -> (block_size, bytes_per_block). bytes/element =
#: bytes_per_block / block_size. Covers the k-quant / legacy / IQ / float
#: types that appear in Unsloth "UD-*_K_XL" and stock GGUF checkpoints; an
#: unknown type falls back to 2 B/element (BF16-equivalent) with a warning.
_GGML_TYPE_SIZE: Dict[int, Tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    30: (1, 2),     # BF16
}

#: GGUF metadata value-type enum (subset used by the header).
_GGUF_T_UINT8, _GGUF_T_INT8, _GGUF_T_UINT16, _GGUF_T_INT16 = 0, 1, 2, 3
_GGUF_T_UINT32, _GGUF_T_INT32, _GGUF_T_FLOAT32, _GGUF_T_BOOL = 4, 5, 6, 7
_GGUF_T_STRING, _GGUF_T_ARRAY, _GGUF_T_UINT64, _GGUF_T_INT64 = 8, 9, 10, 11
_GGUF_T_FLOAT64 = 12
_GGUF_SCALAR_FMT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B",
    10: "Q", 11: "q", 12: "d",
}


def _is_gguf_model(model_path: Optional[str]) -> bool:
    """A GGUF checkpoint is addressed by a single .gguf file rather than a
    directory with config.json."""
    if not model_path:
        return False
    p = str(model_path)
    return p.lower().endswith(".gguf") or os.path.isfile(p)


def _read_gguf_metadata(path: str) -> Tuple[Dict[str, object], Dict[str, int], List[dict]]:
    """Return (scalar KV dict, array-length dict, tensor-info list) from a
    GGUF file's header. Array values are NOT materialized (the token/merge
    arrays hold ~250k entries); only their length is recorded, which is all
    the cost model needs (vocab size). Tensor infos are
    [{name, dims, ggml_type}]."""
    import struct

    def rd(f, fmt: str):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr(f) -> str:
        n = rd(f, "Q")
        return f.read(n).decode("utf-8", "replace")

    def skip_array(f) -> int:
        et = rd(f, "I")
        n = rd(f, "Q")
        if et in _GGUF_SCALAR_FMT:
            f.seek(struct.calcsize(_GGUF_SCALAR_FMT[et]) * n, os.SEEK_CUR)
        elif et == _GGUF_T_STRING:
            for _ in range(n):
                f.seek(rd(f, "Q"), os.SEEK_CUR)
        elif et == _GGUF_T_ARRAY:
            for _ in range(n):
                skip_array(f)
        return n

    def read_value(f, t: int):
        if t in _GGUF_SCALAR_FMT:
            return rd(f, _GGUF_SCALAR_FMT[t])
        if t == _GGUF_T_STRING:
            return rstr(f)
        if t == _GGUF_T_ARRAY:
            return None  # length captured separately
        raise ValueError(f"unsupported GGUF value type {t}")

    scalars: Dict[str, object] = {}
    array_lens: Dict[str, int] = {}
    tensors: List[dict] = []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        rd(f, "I")  # version
        n_tensors = rd(f, "Q")
        n_kv = rd(f, "Q")
        for _ in range(n_kv):
            key = rstr(f)
            t = rd(f, "I")
            if t == _GGUF_T_ARRAY:
                # Peek element type/count without consuming, then skip.
                array_lens[key] = skip_array(f)
            else:
                scalars[key] = read_value(f, t)
        for _ in range(n_tensors):
            name = rstr(f)
            nd = rd(f, "I")
            dims = [rd(f, "Q") for _ in range(nd)]
            gtype = rd(f, "I")
            rd(f, "Q")  # data offset (unused)
            tensors.append({"name": name, "dims": dims, "ggml_type": gtype})
    return scalars, array_lens, tensors


def _gguf_type_bytes_per_elem(gtype: int) -> float:
    entry = _GGML_TYPE_SIZE.get(gtype)
    if entry is None:
        logger.warning(
            "auto-performance: unknown ggml quant type %d in GGUF checkpoint; "
            "assuming 2 B/element for the family byte model.",
            gtype,
        )
        return 2.0
    block, tbytes = entry
    return tbytes / block


def _gguf_family_of(name: str) -> Optional[str]:
    """Map a GGUF tensor name onto the cost model's weight families. The GDN
    (linear-attention) in_proj is stored under ``attn_qkv``/``attn_gate`` in
    llama.cpp's qwen35 naming, so it is grouped with ``ssm_*`` into 'gdn';
    only the full-attention q/k/v/o projections are 'attn'."""
    if "nextn" in name or "mtp" in name:
        return "draft"
    if "ffn_" in name and "exp" not in name:
        return "mlp"
    if name in ("token_embd.weight", "output.weight"):
        return "vocab"
    if "attn_qkv" in name or "attn_gate" in name or "ssm" in name or "conv" in name:
        return "gdn"
    if any(s in name for s in ("attn_q.", "attn_k.", "attn_v.", "attn_output")):
        return "attn"
    return None  # norms / biases (negligible mass) -> ignored in the avg


def _gguf_config_and_families(path: str) -> dict:
    """Synthesize an HF-config-shaped dict for the perf cost model from a
    GGUF header, plus measured per-family bytes/parameter and a few private
    hint keys (prefixed ``__gguf_``). Raises a clear error if a required
    field is missing instead of crashing later deep in the math."""
    scalars, array_lens, tensors = _read_gguf_metadata(path)
    arch = scalars.get("general.architecture")
    if not arch:
        raise ValueError(f"{path}: GGUF header lacks general.architecture")

    def need(key: str):
        full = f"{arch}.{key}"
        if full not in scalars:
            raise ValueError(
                f"auto-performance: GGUF checkpoint {path} is missing the "
                f"required metadata key '{full}'; cannot build the perf cost "
                f"model. Pin --rank-mlp-ratio to skip the optimizer."
            )
        return scalars[full]

    def opt(key: str, default):
        return scalars.get(f"{arch}.{key}", default)

    hidden = int(need("embedding_length"))
    intermediate = int(need("feed_forward_length"))
    q_heads = int(need("attention.head_count"))
    kv_heads = int(need("attention.head_count_kv"))
    head_dim = int(opt("attention.key_length", hidden // max(q_heads, 1)))
    block_count = int(need("block_count"))
    nextn = int(opt("nextn_predict_layers", 0) or 0)
    n_layers = block_count - nextn  # nextn/MTP block is not a base layer
    interval = int(opt("full_attention_interval", 0) or 0)
    if interval > 0:
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(n_layers)
        ]
    else:
        layer_types = None  # __init__ falls back to all-full

    # Vocab size = length of the token list (arrays are not materialized).
    vocab = (
        array_lens.get("tokenizer.ggml.tokens")
        or array_lens.get("tokenizer.ggml.token_type")
        or int(opt("vocab_size", 0) or 0)
    )

    # GDN / SSM geometry (llama.cpp ssm.* keys).
    gdn_k_heads = int(opt("ssm.group_count", 0) or 0)
    gdn_v_heads = int(opt("ssm.time_step_rank", 0) or 0)
    gdn_k_dim = int(opt("ssm.state_size", 0) or 0)
    ssm_inner = int(opt("ssm.inner_size", 0) or 0)
    gdn_v_dim = (ssm_inner // gdn_v_heads) if gdn_v_heads else gdn_k_dim
    conv_kernel = int(opt("ssm.conv_kernel", 4) or 4)

    # Per-family bytes/element from the tensor quant types (element-weighted).
    fam_bytes: Dict[str, float] = {}
    fam_elems: Dict[str, float] = {}
    attn_q_out = 0
    has_draft_body = False
    for t in tensors:
        fam = _gguf_family_of(t["name"])
        dims = t["dims"]
        elems = 1
        for d in dims:
            elems *= d
        if "attn_q." in t["name"] and len(dims) >= 2:
            attn_q_out = max(attn_q_out, max(dims))
        if "nextn" in t["name"] and any(
            s in t["name"] for s in ("attn_q", "attn_k", "attn_v", "ffn_")
        ):
            has_draft_body = True
        if fam is None:
            continue
        bpe = _gguf_type_bytes_per_elem(t["ggml_type"])
        fam_bytes[fam] = fam_bytes.get(fam, 0.0) + elems * bpe
        fam_elems[fam] = fam_elems.get(fam, 0.0) + elems
    family_bpp = {
        fam: (fam_bytes[fam] / fam_elems[fam]) for fam in fam_bytes if fam_elems[fam]
    }
    # Reasonable fallbacks if a family had no matched tensors.
    family_bpp.setdefault("attn", 1.0)
    family_bpp.setdefault("mlp", 0.75)
    family_bpp.setdefault("gdn", 2.0)
    family_bpp.setdefault("vocab", 1.0625)
    family_bpp.setdefault("draft", 1.0625)

    # attn_output_gate: the full-attention q projection emits q + output gate
    # (2x q_heads*head_dim) when gating is on. Detect from the tensor shape;
    # default False if no separate attn_q tensor was found.
    attn_gate = bool(attn_q_out >= 2 * q_heads * head_dim and attn_q_out > 0)

    # MLP shard granularity: the down-proj is quantized in blocks along the
    # contracted (intermediate) axis, so the natural indivisible unit is the
    # dominant MLP quant block (K-quants = 256, Q8_0 = 32).
    mlp_types = [t["ggml_type"] for t in tensors if _gguf_family_of(t["name"]) == "mlp"]
    if mlp_types:
        dominant = max(set(mlp_types), key=mlp_types.count)
        mlp_group = _GGML_TYPE_SIZE.get(dominant, (256, 0))[0]
    else:
        mlp_group = 256

    return {
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": n_layers,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "attn_output_gate": attn_gate,
            "vocab_size": int(vocab),
            "mtp_num_hidden_layers": nextn,
            "linear_num_key_heads": gdn_k_heads,
            "linear_num_value_heads": gdn_v_heads,
            "linear_key_head_dim": gdn_k_dim,
            "linear_value_head_dim": gdn_v_dim,
            "linear_conv_kernel_dim": conv_kernel,
            "layer_types": layer_types,
        },
        "quantization_config": {"group_size": mlp_group},
        "__gguf_family_bpp__": family_bpp,
        "__gguf_has_draft_body__": has_draft_body,
    }


class PerfCostModel:
    """Parse-time capacity/speed predictor for MLP-vector candidates.

    All quantities are derived from config.json + the on-disk checkpoint
    size + the resolved --rank-tp-ratio auto budgets, BEFORE any weights are
    loaded. Absolute token numbers are estimates (logged as such); the floor
    decision only consumes candidate-over-base RATIOS, whose dominant term
    (MLP bytes per unit) is exact.
    """

    def __init__(self, plan_inputs, base_plan: List[int], budgets_mib: List[int]):
        # ``plan_inputs`` is a PlanInputs dataclass (see above). The boot
        # path builds it via PlanInputs.from_server_args so the server and
        # the offline planner (sglang.srt.planner) run identical sizing.
        self.tp_size = plan_inputs.tp_size
        self.base_plan = list(base_plan)
        self.budgets_mib = list(budgets_mib)
        self.plan_inputs = plan_inputs

        cfg = self._load_config(plan_inputs.model_path)
        text = cfg.get("text_config", cfg)
        self.hidden = int(text["hidden_size"])
        self.intermediate = int(text["intermediate_size"])
        layer_types = text.get("layer_types")
        n_layers = int(text["num_hidden_layers"])
        if layer_types:
            self.full_layers = sum(1 for t in layer_types if t == "full_attention")
            self.gdn_layers = sum(1 for t in layer_types if t == "linear_attention")
        else:
            self.full_layers, self.gdn_layers = n_layers, 0
        self.n_layers = n_layers
        self.kv_heads = int(text.get("num_key_value_heads", 1))
        self.q_heads = int(text.get("num_attention_heads", 1))
        self.head_dim = int(text.get("head_dim", self.hidden // self.q_heads))
        self.attn_gate = bool(text.get("attn_output_gate", False))
        self.vocab = int(text.get("vocab_size", 0))
        self.gdn_k_heads = int(text.get("linear_num_key_heads", 0) or 0)
        self.gdn_v_heads = int(text.get("linear_num_value_heads", 0) or 0)
        self.gdn_k_dim = int(text.get("linear_key_head_dim", 0) or 0)
        self.gdn_v_dim = int(text.get("linear_value_head_dim", 0) or 0)
        self.conv_kernel = int(text.get("linear_conv_kernel_dim", 4) or 4)
        self.mtp_layers = int(text.get("mtp_num_hidden_layers", 0) or 0)

        # Unit grids (must match the model's tp_units so candidate vectors
        # materialize identically to the real partition).
        self.attn_units = max(self.kv_heads, 1)
        self.gdn_units = max(self.gdn_k_heads, 1)
        quant_cfg = cfg.get("quantization_config") or {}
        group = self._quant_group_size(quant_cfg)
        if group and self.intermediate % group == 0:
            self.mlp_units = self.intermediate // group
        elif self.intermediate % 128 == 0:
            self.mlp_units = self.intermediate // 128
        else:
            self.mlp_units = math.gcd(self.intermediate, 512) or 1

        self.spec_active = plan_inputs.speculative_algorithm is not None
        self.spec_draft_tokens = int(plan_inputs.speculative_num_draft_tokens or 0)
        kv_dtype = str(plan_inputs.kv_cache_dtype or "auto")
        self.kv_cell_bytes_per_layer = (
            2 * self.kv_heads * self.head_dim * (1 if "fp8" in kv_dtype else 2)
        )
        cell_layers = self.full_layers + (self.mtp_layers if self.spec_active else 0)
        #: Full-kv-head KV bytes per token (weighted DCP replicates heads).
        self.kv_cell_bytes = self.kv_cell_bytes_per_layer * cell_layers

        self.families = self._build_families(cfg)
        self.mamba_pool_bytes = self._mamba_pool_bytes()

    @staticmethod
    def _load_config(model_path: str) -> dict:
        # GGUF checkpoints are a single .gguf FILE, not a directory with a
        # config.json -- read the fields (and per-family quant bytes) from the
        # GGUF header instead of crashing on open(model_path/'config.json').
        if _is_gguf_model(model_path):
            return _gguf_config_and_families(model_path)
        with open(os.path.join(model_path, "config.json")) as f:
            return json.load(f)

    @staticmethod
    def _quant_group_size(quant_cfg: dict) -> Optional[int]:
        groups = quant_cfg.get("config_groups") or {}
        for g in groups.values():
            w = (g or {}).get("weights") or {}
            gs = w.get("group_size")
            if gs:
                return int(gs)
        gs = quant_cfg.get("group_size")
        return int(gs) if gs else None

    def _build_families(self, cfg: dict) -> Dict[str, _Family]:
        H, I = self.hidden, self.intermediate
        q_size = self.q_heads * self.head_dim * (2 if self.attn_gate else 1)
        kv_size = self.kv_heads * self.head_dim
        attn_layer = H * (q_size + 2 * kv_size) + (self.q_heads * self.head_dim) * H
        mlp_layer = 3 * H * I

        gdn_layer = 0.0
        if self.gdn_layers:
            k_sz = self.gdn_k_heads * self.gdn_k_dim
            v_sz = self.gdn_v_heads * self.gdn_v_dim
            # in_proj (q,k,v,z) + b/a + out_proj + conv + norms (approx).
            gdn_layer = (
                H * (2 * k_sz + 2 * v_sz)
                + H * 2 * self.gdn_v_heads
                + v_sz * H
                + (2 * k_sz + v_sz) * self.conv_kernel
            )

        vocab_params = 2.0 * self.vocab * H  # embed + lm_head (untied worst case)

        vision_params = 0.0
        vcfg = cfg.get("vision_config")
        if vcfg and not cfg.get("language_model_only", False):
            vh = int(vcfg.get("hidden_size", 0) or 0)
            vi = int(vcfg.get("intermediate_size", 0) or 0)
            vd = int(vcfg.get("depth", 0) or 0)
            vision_params = vd * (4 * vh * vh + 2 * vh * vi)

        draft_attn = draft_mlp = draft_repl = 0.0
        if self.spec_active and self.mtp_layers:
            draft_attn = self.mtp_layers * attn_layer
            draft_mlp = self.mtp_layers * mlp_layer
            draft_repl = 2 * H * H  # fc (2H -> H), bf16, replicated
            # NOTE eb764a12b: the draft's embed/lm_head duplicates are shared
            # with the target BEFORE KV profiling, so they are a load-time
            # transient only and deliberately NOT part of this budget model.

        families = {
            "attn": _Family(self.full_layers * attn_layer, 2.0, "attn"),
            "gdn": _Family(self.gdn_layers * gdn_layer, 2.0, "gdn"),
            "mlp": _Family(self.n_layers * mlp_layer, 2.0, "mlp"),
            "vocab": _Family(vocab_params, 2.0, "even"),
            "vision": _Family(vision_params, 2.0, "gdn_base"),
            "draft_attn": _Family(draft_attn, 2.0, "attn"),
            "draft_mlp": _Family(draft_mlp, 2.0, "mlp"),
            "draft_repl": _Family(draft_repl, 2.0, "replicated"),
        }

        # GGUF path: every family (embed/lm_head, attention, MLP, GDN) is
        # quantized on its own ggml grid, so instead of the safetensors
        # "BF16-inside-INT4" anchoring below we set each family's bytes/param
        # directly from the measured per-family quant types (element-weighted
        # bytes/element read from the tensor-info block). Assumptions: norms/
        # biases are folded into their family's average (negligible mass); the
        # GDN in_proj (llama.cpp ``attn_qkv``) is counted in 'gdn'; when the
        # nextn/MTP block carries no own attn/ffn weights (module sharing) its
        # draft_attn/draft_mlp mass is zeroed so it is not double-counted.
        gguf_bpp = cfg.get("__gguf_family_bpp__")
        if gguf_bpp is not None:
            families["attn"].bytes_per_param = gguf_bpp["attn"]
            families["mlp"].bytes_per_param = gguf_bpp["mlp"]
            families["gdn"].bytes_per_param = gguf_bpp["gdn"]
            families["vocab"].bytes_per_param = gguf_bpp["vocab"]
            families["draft_repl"].bytes_per_param = gguf_bpp.get("draft", 2.0)
            families["draft_attn"].bytes_per_param = gguf_bpp["attn"]
            families["draft_mlp"].bytes_per_param = gguf_bpp["mlp"]
            if not cfg.get("__gguf_has_draft_body__", False):
                families["draft_attn"].params = 0.0
                families["draft_mlp"].params = 0.0
            return families

        # Anchor quantized-family bytes/param on the measured checkpoint
        # size: BF16 families (GDN, embed/lm_head, vision, draft fc -- the
        # "BF16 inside INT4" cost-model term) stay at 2 B/param, the
        # remaining checkpoint bytes are spread over the quantized families
        # (attn + MLP + draft layer) proportionally to their param counts.
        from sglang.srt.distributed.utils import _checkpoint_size_mib

        ckpt_bytes = _checkpoint_size_mib(self.plan_inputs.model_path) * 2**20
        quant_names = ("attn", "mlp", "draft_attn", "draft_mlp")
        bf16_bytes = sum(
            fam.bytes for name, fam in families.items() if name not in quant_names
        )
        quant_params = sum(families[name].params for name in quant_names)
        if ckpt_bytes > 0 and quant_params > 0:
            bpp = (ckpt_bytes - bf16_bytes) / quant_params
            bpp = min(max(bpp, 0.5), 2.25)  # int4+scales ... bf16 bounds
            for name in quant_names:
                families[name].bytes_per_param = bpp
        return families

    def _shard_fractions(self, shard: str, mlp_vector: List[int]) -> List[float]:
        from sglang.srt.distributed.utils import partition_units

        n = self.tp_size
        if shard == "even":
            return [1.0 / n] * n
        if shard == "replicated":
            return [1.0] * n
        if shard == "attn":
            units = partition_units(self.attn_units, self.base_plan)
            return [u / self.attn_units for u in units]
        if shard in ("gdn", "gdn_base"):
            # vision ("gdn_base") has no own family vector; it follows the
            # base plan on a fine grid -> approximate with exact proportion.
            if shard == "gdn" and self.gdn_units >= n:
                units = partition_units(self.gdn_units, self.base_plan)
                return [u / self.gdn_units for u in units]
            total = float(sum(self.base_plan))
            return [w / total for w in self.base_plan]
        if shard == "mlp":
            units = partition_units(self.mlp_units, mlp_vector)
            return [u / self.mlp_units for u in units]
        raise ValueError(shard)

    def mlp_unit_partition(self, mlp_vector: List[int]) -> List[int]:
        from sglang.srt.distributed.utils import partition_units

        return partition_units(self.mlp_units, mlp_vector)

    def gdn_unit_partition(self) -> List[int]:
        from sglang.srt.distributed.utils import partition_units

        if self.gdn_units >= self.tp_size:
            return partition_units(self.gdn_units, self.base_plan)
        return [0] * self.tp_size

    def per_rank_weight_bytes(self, mlp_vector: List[int]) -> List[float]:
        totals = [0.0] * self.tp_size
        for fam in self.families.values():
            if fam.params <= 0:
                continue
            fracs = self._shard_fractions(fam.shard, mlp_vector)
            for r in range(self.tp_size):
                totals[r] += fam.bytes * fracs[r]
        return totals

    def _mamba_pool_bytes(self) -> List[float]:
        """Per-rank mamba/SSM pool bytes (state pool + spec-decode
        intermediate), the M22 "SSM pool moves with GDN units" term. Sized
        like the auto-mamba demand path: slots = ceil(target * ratio * 1.25),
        per-request state scales with the rank's GDN-unit share."""
        if not self.gdn_layers or not self.gdn_units:
            return [0.0] * self.tp_size
        ssm_env = os.environ.get("SGLANG_MAMBA_SSM_DTYPE", "")
        ssm_bytes = 2 if "bfloat16" in ssm_env or "float16" in ssm_env else 4
        heads_per_unit = max(self.gdn_v_heads // max(self.gdn_k_heads, 1), 1)
        state_per_unit_layer = (
            heads_per_unit * self.gdn_v_dim * self.gdn_k_dim * ssm_bytes
        )
        conv_per_unit_layer = (
            (2 * self.gdn_k_dim + heads_per_unit * self.gdn_v_dim)
            * (self.conv_kernel - 1)
            * 2
        )
        per_req_per_unit = self.gdn_layers * (
            state_per_unit_layer + conv_per_unit_layer
        )

        target = self.plan_inputs.max_running_requests or 16
        target = min(target, 48)
        ratio = 5  # MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO(3) + overlap(2)
        slots = math.ceil(target * ratio * 1.25)
        d = self.spec_draft_tokens if self.spec_active else 0
        eff_slots = slots + min(target, slots // ratio) * d

        gdn_units = self.gdn_unit_partition()
        return [per_req_per_unit * u * eff_slots for u in gdn_units]

    # -- capacity prediction ------------------------------------------------

    def predict_capacity(self, mlp_vector: List[int]) -> dict:
        """Predicted per-rank KV token capacity P_r and max context for a
        candidate MLP vector (the base plan's attention/GDN/DCP splits are
        held fixed -- decode is flat across splits per M22, so they are not
        levers). Same math family as the pool sizing: budget minus weight
        bytes minus mamba pool minus reserves, divided by the full-kv-head
        cell; context C = min_r(P_r/ratio_r) * sum(ratios), whose converged
        optimum is min(sum_r P_r, 64 * min_r P_r)."""
        from sglang.srt.distributed.utils import partition_units

        weights = self.per_rank_weight_bytes(mlp_vector)
        overhead = (
            _PREDICT_OVERHEAD_MIB + _PREDICT_MAMBA_ACT_RESERVE_MIB
        ) * 2**20
        p = []
        for r in range(self.tp_size):
            budget = self.budgets_mib[r] * 2**20
            free = budget - weights[r] - self.mamba_pool_bytes[r] - overhead
            p.append(free / self.kv_cell_bytes)
        feasible = all(x >= _PREDICT_MIN_RANK_TOKENS for x in p)
        if feasible:
            ctx = min(sum(p), _PREDICT_TOKEN_UNITS * min(p))
            vec = partition_units(
                _PREDICT_TOKEN_UNITS, [max(int(x), 1) for x in p]
            )
            g = math.gcd(*vec)
            vec = [v // g for v in vec]
        else:
            ctx, vec = 0.0, None
        return {
            "p": p,
            "ctx": ctx,
            "token_vector": vec,
            "feasible": feasible,
            "weights_gib": [w / 2**30 for w in weights],
        }

    # -- speed prediction ---------------------------------------------------

    def streamed_bytes(self, mlp_vector: List[int]) -> List[float]:
        """Per-rank weight bytes STREAMED per decode token (bs=1 decode is
        weight-bandwidth-bound; replicated families stream on every rank)."""
        totals = [0.0] * self.tp_size
        for fam in self.families.values():
            if fam.params <= 0:
                continue
            fracs = self._shard_fractions(fam.shard, mlp_vector)
            for r in range(self.tp_size):
                totals[r] += fam.bytes * fracs[r]
        return totals

    def _mlp_unit_share(self, total_streamed: float) -> float:
        """Streamed-byte-share contributed by ONE MLP unit (the granularity
        of a single representable concentration step). Used to size the
        coarse-grid headroom in the decode-knee guard."""
        if total_streamed <= 0 or self.mlp_units <= 0:
            return 0.0
        return (self.families["mlp"].bytes / self.mlp_units) / total_streamed

    def decode_knee_ok(
        self,
        mlp_vector: List[int],
        membw_gbs: List[float],
        tol: float = _PREDICT_DECODE_KNEE_TOL,
    ) -> bool:
        """Decode-knee guard (M20/M22/M23-measured): decode throughput is
        FLAT while every rank's share of the streamed weight bytes stays at
        or below its share of the rig's total memory bandwidth; pushing a
        rank past that knee makes it the decode lockstep bottleneck (M23:
        16,1,2 = 56.5% bytes on the 51.9%-membw 5090 -> decode -4.8%) and
        the prefill model's extra predicted gain does NOT materialize
        (measured +10.0%, identical to the knee-exact C6 vector).

        The per-rank share is computed from the ACTUAL integer unit partition
        (``mlp_unit_partition`` via ``streamed_bytes``), never from the wish
        ratio, so a coarse grid's rounding is taken at face value. Because
        coarse grids (FP8 dense ~136 units, GGUF K-quant ~68) realize the
        split in large steps, the nearest representable vector can sit above
        the measured knee while its whole-model share still reads just under
        the bandwidth share (M27d: FP8 4,1,1 = 51.5% share < 51.9% membw, yet
        decode -14/-24%). On such coarse grids we therefore require
        ``_PREDICT_KNEE_COARSE_HEADROOM_UNITS`` unit-steps of headroom below
        the knee and round DOWN to the last safe vector; fine grids keep the
        exact bandwidth-share ceiling. A rank that only sheds/keeps bytes vs
        the base plan can never become the knee and is skipped."""
        ok, _ = self.decode_knee_detail(mlp_vector, membw_gbs)
        return ok

    def decode_knee_detail(
        self, mlp_vector: List[int], membw_gbs: List[float]
    ) -> Tuple[bool, Optional[str]]:
        """Like ``decode_knee_ok`` but also returns a human-readable reason
        for the first violating rank, naming the unit granularity (for the
        optimizer's per-candidate log line)."""
        cand = self.streamed_bytes(mlp_vector)
        base = self.streamed_bytes(self.base_plan)
        total = sum(cand)
        bw_total = sum(membw_gbs)
        coarse = self.mlp_units < _PREDICT_KNEE_COARSE_UNITS
        unit_share = self._mlp_unit_share(total)
        headroom = (
            _PREDICT_KNEE_COARSE_HEADROOM_UNITS * unit_share if coarse else 0.0
        )
        for r in range(self.tp_size):
            if cand[r] <= base[r] * (1.0 + 1e-6):
                continue  # rank sheds or keeps bytes: cannot become the knee
            achievable = cand[r] / total if total else 0.0
            membw_share = membw_gbs[r] / bw_total if bw_total else 0.0
            limit = membw_share - headroom
            if achievable > limit:
                # requested = share the wish ratio (continuous, no rounding)
                # aimed for on this rank; achievable = the real partition's
                # share; naming both makes the coarse-grid overshoot explicit.
                requested = self._requested_mlp_share(mlp_vector, r, total)
                grain = (
                    f"coarse {self.mlp_units}-unit MLP grid, "
                    f"{_PREDICT_KNEE_COARSE_HEADROOM_UNITS}-unit headroom"
                    if coarse
                    else f"fine {self.mlp_units}-unit MLP grid"
                )
                reason = (
                    f"unit granularity ({grain}): rank {r} requested "
                    f"{requested * 100:.1f}% -> achievable {achievable * 100:.1f}% "
                    f"of streamed weight bytes exceeds membw share "
                    f"{membw_share * 100:.1f}% (safe ceiling {limit * 100:.1f}%)"
                )
                return False, reason
        return True, None

    def _requested_mlp_share(
        self, mlp_vector: List[int], rank: int, total_streamed: float
    ) -> float:
        """Whole-model streamed-byte share rank `rank` would have if the MLP
        family used the CONTINUOUS wish fraction (no integer unit rounding).
        Contrasted with the achievable (real-partition) share to expose the
        coarse-grid overshoot in the log."""
        if total_streamed <= 0:
            return 0.0
        wish = mlp_vector[rank] / sum(mlp_vector)
        acc = 0.0
        for name, fam in self.families.items():
            if fam.params <= 0:
                continue
            if fam.shard == "mlp":
                acc += fam.bytes * wish
            else:
                acc += fam.bytes * self._shard_fractions(fam.shard, mlp_vector)[rank]
        return acc / total_streamed

    def prefill_time_model(
        self, mlp_vector: List[int], gemm_tflops: List[float], min_link_gbs: float
    ) -> float:
        """Relative prefill step time: lockstep compute max over ranks
        (per-token flops ~ 2 x sharded params, the param-proxy) plus the
        ring-all-reduce term over the narrowest link. Only ratios between
        candidates are consumed."""
        t_comp = 0.0
        for r in range(self.tp_size):
            params_r = 0.0
            for fam in self.families.values():
                if fam.params <= 0 or fam.shard == "replicated":
                    continue
                params_r += (
                    fam.params * self._shard_fractions(fam.shard, mlp_vector)[r]
                )
            t = 2.0 * params_r / (gemm_tflops[r] * 1e12 * _PREDICT_GEMM_EFF)
            t_comp = max(t_comp, t)
        # Two all-reduces of H bf16 per layer per token, ring factor
        # 2(N-1)/N, bounded by the narrowest participating link.
        n = self.tp_size
        ar_bytes = self.n_layers * 2 * self.hidden * 2
        t_comm = (
            ar_bytes * 2 * (n - 1) / n / (max(min_link_gbs, 0.1) * 1e9)
            if n > 1
            else 0.0
        )
        return t_comp + t_comm


# ---------------------------------------------------------------------------
# The optimizer: candidate MLP vectors -> floor filter -> objective.
# ---------------------------------------------------------------------------


def _gcd_reduce(vec: Sequence[int]) -> Tuple[int, ...]:
    g = math.gcd(*vec)
    return tuple(v // g for v in vec)


def _mlp_candidates(
    model: PerfCostModel, scores: List[float], base_plan: List[int]
) -> List[Tuple[int, ...]]:
    """Small deduplicated candidate set of MLP unit vectors.

    Two ladders: (a) score-proportional concentration at several strengths
    (score^alpha), (b) the compute-balance solution (assign each rank MLP
    mass until work_r/score_r equalizes, the analytic prefill optimum),
    rounded at several integer resolutions."""
    n = model.tp_size
    cands: List[Tuple[int, ...]] = []
    smax = max(scores)

    for alpha in (0.5, 1.0, 1.5, 2.0):
        for k in (3, 5, 8):
            vec = tuple(
                max(1, round(k * (s / smax) ** alpha)) for s in scores
            )
            cands.append(_gcd_reduce(vec))

    # Balance ladder: fixed (non-MLP) work per rank + m_r ~ score share.
    fixed = [0.0] * n
    mlp_mass = 0.0
    for fam in model.families.values():
        if fam.params <= 0 or fam.shard == "replicated":
            continue
        if fam.shard == "mlp":
            mlp_mass += fam.params
            continue
        fr = model._shard_fractions(fam.shard, base_plan)
        for r in range(n):
            fixed[r] += fam.params * fr[r]
    total = sum(fixed) + mlp_mass
    s_sum = sum(scores)
    m = [max(total * s / s_sum - fixed[r], 0.0) for r, s in enumerate(scores)]
    m_sum = sum(m)
    if m_sum > 0:
        for k in (6, 10, 16):
            vec = tuple(max(1, round(k * x / max(m))) for x in m)
            cands.append(_gcd_reduce(vec))

    base = _gcd_reduce(base_plan)
    out, seen = [], set()
    for c in cands:
        if c not in seen and c != base and len(set(c)) > 1:
            seen.add(c)
            out.append(c)
    return out


@dataclasses.dataclass
class PerfDecision:
    chosen_vector: Optional[List[int]]  # None = keep the plain auto split
    log_lines: List[str]


def _tp_drop_recommendation(
    server_args, profile: dict, gpus: List[dict], model: PerfCostModel
) -> Optional[str]:
    """Stage-1 TP-degree reduction RECOMMENDATION (log only, never applied):
    when one GPU sits behind a clearly narrower link and the remaining
    budgets still fit the weights, dropping it bought +55-76% prefill /
    +25-30% concurrent at -72% context in the M22 measurements."""
    if server_args.tp_size < 3 or not profile.get("links"):
        return None
    by_uuid = {g["uuid"]: g for g in gpus}
    rank_gpu = server_args.rank_gpu_id
    used_uuids = []
    for gid in rank_gpu:
        match = [g for g in gpus if g["cuda_index"] == gid]
        if not match:
            return None
        used_uuids.append(match[0]["uuid"])
    unique = sorted(set(used_uuids))
    if len(unique) < 3:
        return None

    def pair_bw(u1: str, u2: str) -> Optional[float]:
        e = profile["links"].get("|".join(sorted([u1, u2])))
        return e.get("p2p_gbs") if e else None

    best_link = 0.0
    per_gpu_best: Dict[str, float] = {}
    for u in unique:
        bws = [pair_bw(u, v) for v in unique if v != u]
        bws = [b for b in bws if b]
        if not bws:
            return None
        per_gpu_best[u] = max(bws)
        best_link = max(best_link, max(bws))
    weakest = min(per_gpu_best, key=per_gpu_best.get)
    if per_gpu_best[weakest] >= _TP_DROP_LINK_FRACTION * best_link:
        return None

    # Fit check: total weights (even split irrelevant -- total conserved)
    # must fit into the REMAINING ranks' budgets with headroom.
    total_weights = sum(model.per_rank_weight_bytes(list(model.base_plan)))
    keep_budget = sum(
        model.budgets_mib[r] * 2**20
        for r in range(model.tp_size)
        if used_uuids[r] != weakest
    )
    if total_weights > _TP_DROP_FIT_FACTOR * keep_budget:
        return None

    keep_ids = [
        str(rank_gpu[r]) for r in range(model.tp_size) if used_uuids[r] != weakest
    ]
    name = by_uuid[weakest]["name"]
    return (
        f"RECOMMENDATION (not applied): GPU {by_uuid[weakest]['cuda_index']} "
        f"({name}) sits behind the narrowest link "
        f"({per_gpu_best[weakest]:.1f} GB/s vs best {best_link:.1f} GB/s). "
        f"Dropping it (--tp-size {len(keep_ids)} --rank-gpu-id "
        f"{','.join(keep_ids)}) measured +55-76% prefill / +25-30% "
        "concurrent at -72% max context in the M22 feasibility matrix. "
        "TP degree is never changed silently; re-launch with those flags "
        "to take this trade."
    )


def apply_auto_performance(server_args) -> None:
    """Entry point for --rank-tp-ratio auto-performance, called from
    ServerArgs._handle_uneven_tp AFTER the VRAM-auto base split is resolved
    and BEFORE the family-vector validation. Derives --rank-mlp-ratio from
    the hardware profile, subject to the context floor; logs one block with
    every decision input. Never touches the base (attention/GDN/DCP) split.
    """
    lines: List[str] = ["auto-performance (--rank-tp-ratio auto-performance):"]

    def emit():
        logger.info("\n  ".join(lines))

    tune = server_args.rank_perf_tune
    loose = float(server_args.rank_perf_loose_ctx_percent)

    if not isinstance(server_args.rank_tp_ratio, list):
        lines.append(
            "base VRAM-auto split collapsed to the even split (uniform "
            "budgets); the MLP family vector requires an uneven base plan "
            "-- keeping the classic even split unchanged."
        )
        emit()
        return

    base_plan = list(server_args.rank_tp_ratio)
    budgets = server_args.rank_gpu_memory_mib
    budgets = (
        list(budgets)
        if isinstance(budgets, list)
        else [budgets] * server_args.tp_size
    )

    # Pin path: an explicit vector (flag or env) skips probe + optimizer.
    from sglang.srt.environ import envs

    pinned = server_args.rank_mlp_ratio
    env_pin = envs.SGLANG_UNEVEN_MLP_VECTOR.get()
    if pinned is not None or env_pin:
        lines.append(
            f"MLP vector PINNED ({'SGLANG_UNEVEN_MLP_VECTOR=' + env_pin if env_pin else '--rank-mlp-ratio ' + ','.join(map(str, pinned))}); "
            "hardware probe and optimizer skipped (pin path)."
        )
        emit()
        return

    profile, source, gpus = get_hardware_profile()
    if profile is None:
        lines.append(
            "hardware profile unavailable (probe failed) -- keeping the "
            "plain VRAM-auto split; fix the probe or pin --rank-mlp-ratio."
        )
        emit()
        return

    lines.append(f"hardware profile: {source}")
    uuid_by_idx = {g["cuda_index"]: g["uuid"] for g in gpus}
    rank_scores_gemm: List[float] = []
    rank_scores_bw: List[float] = []
    rank_names: List[str] = []
    for gid in server_args.rank_gpu_id:
        entry = profile["gpus"].get(uuid_by_idx.get(gid, ""), None)
        if entry is None:
            lines.append(
                f"GPU {gid} missing from the profile -- keeping plain auto."
            )
            emit()
            return
        rank_scores_gemm.append(entry["gemm_tflops"])
        rank_scores_bw.append(entry["membw_gbs"])
        rank_names.append(entry["name"])
    for r in range(server_args.tp_size):
        lines.append(
            f"rank {r} -> GPU {server_args.rank_gpu_id[r]} ({rank_names[r]}): "
            f"GEMM {rank_scores_gemm[r]:.1f} TFLOPS, "
            f"membw {rank_scores_bw[r]:.0f} GB/s"
        )
    links = profile.get("links") or {}
    pair_bws = [v["p2p_gbs"] for k, v in links.items() if k != "__group__"]
    min_link = min(pair_bws) if pair_bws else 8.0
    if pair_bws:
        lines.append(
            "link matrix: "
            + ", ".join(
                f"{k.split('|')[0][-8:]}<->{k.split('|')[1][-8:]} "
                f"{v['p2p_gbs']:.1f} GB/s"
                for k, v in links.items()
                if k != "__group__"
            )
        )

    model = PerfCostModel(
        PlanInputs.from_server_args(server_args), base_plan, budgets
    )
    base_pred = model.predict_capacity(base_plan)
    floor = (1.0 - loose / 100.0) * base_pred["ctx"]
    lines.append(
        f"VRAM-auto reference: predicted per-rank capacity "
        f"{[int(x) for x in base_pred['p']]} tokens, predicted max context "
        f"~{int(base_pred['ctx'])} (converged weighted-DCP optimum; "
        f"estimate), materialized MLP units "
        f"{model.mlp_unit_partition(base_plan)}"
    )
    lines.append(
        f"context floor (--rank-perf-loose-ctx-percent {loose:g}): "
        f"candidates must predict >= {int(floor)} tokens "
        f"({100 - loose:g}% of the VRAM-auto prediction)"
    )

    # Tuning target (per M22: decode is flat across representable splits;
    # prefill/aggregate is the lever, and 'both' rides the same lever).
    chosen: Optional[Tuple[int, ...]] = None
    if tune == "dec":
        lines.append(
            "tune=dec: M22 measured decode as FLAT (+-2%) across all "
            "representable splits -- the VRAM-auto split already sits near "
            "the decode bandwidth optimum, and over-concentration makes the "
            "strong card the lockstep bottleneck (-6-8%). auto-performance "
            "therefore keeps the auto split unchanged (documented no-op; "
            "only free gains apply, and none exceed auto for decode). Use "
            "--rank-perf-tune enc|both for the measured prefill/throughput "
            "lever."
        )
    else:
        if tune == "both":
            lines.append(
                "tune=both: prefill and concurrent throughput ride the same "
                "MLP-concentration lever (M22: +10% prefill / +7% conc-8 "
                "for the C6-class vector), so 'both' optimizes the same "
                "objective as 'enc'."
            )
        # enc/both objective: minimize the lockstep prefill time model.
        enc_scores = list(rank_scores_gemm)
        # Per-rank link penalty: a rank behind a narrow link attracts fewer
        # units (folded softly; the AR term already carries the group cost).
        if pair_bws and len(set(server_args.rank_gpu_id)) == server_args.tp_size:
            per_rank_link = []
            for gid in server_args.rank_gpu_id:
                u = uuid_by_idx[gid]
                bws = [
                    v["p2p_gbs"]
                    for k, v in links.items()
                    if k != "__group__" and u in k
                ]
                per_rank_link.append(max(bws) if bws else min_link)
            best = max(per_rank_link)
            enc_scores = [
                s * (l / best) ** _PREDICT_LINK_ALPHA
                for s, l in zip(enc_scores, per_rank_link)
            ]
        t_base = model.prefill_time_model(base_plan, enc_scores, min_link)
        candidates = _mlp_candidates(model, enc_scores, base_plan)
        best_gain = 0.0
        results = []
        for cand in candidates:
            pred = model.predict_capacity(list(cand))
            t_cand = model.prefill_time_model(list(cand), enc_scores, min_link)
            gain = t_base / t_cand - 1.0
            knee_ok, knee_reason = model.decode_knee_detail(
                list(cand), rank_scores_bw
            )
            floor_ok = pred["feasible"] and pred["ctx"] >= floor
            results.append((cand, pred, gain, floor_ok, knee_ok, knee_reason))
            if floor_ok and knee_ok and gain > best_gain + 1e-9:
                best_gain = gain
                chosen = cand
        for cand, pred, gain, floor_ok, knee_ok, knee_reason in sorted(
            results, key=lambda x: -x[2]
        )[:6]:
            if not pred["feasible"]:
                verdict = "INFEASIBLE"
            elif not floor_ok:
                verdict = "REJECTED by floor"
            elif not knee_ok:
                verdict = f"REJECTED by decode-knee guard -- {knee_reason}"
            else:
                verdict = "floor OK, knee OK"
            lines.append(
                f"candidate MLP vector {','.join(map(str, cand))}: "
                f"predicted ctx ~{int(pred['ctx'])} ({verdict}), "
                f"predicted prefill gain {gain * 100:+.1f}% "
                f"(units {model.mlp_unit_partition(list(cand)) if pred['feasible'] else 'n/a'})"
            )
        if chosen is None:
            lines.append(
                "no candidate beats the VRAM-auto split within the context "
                "floor + decode-knee guard -- keeping plain auto (floor/knee "
                "binds or no predicted gain; raise "
                "--rank-perf-loose-ctx-percent to trade context for speed "
                "via TP-degree reduction, which stage 1 only recommends)."
            )

    if chosen is not None:
        pred = model.predict_capacity(list(chosen))
        server_args.rank_mlp_ratio = list(chosen)
        lines.append(
            f"CHOSEN MLP vector: {','.join(map(str, chosen))} "
            f"(materialized units {model.mlp_unit_partition(list(chosen))}; "
            f"predicted ctx ~{int(pred['ctx'])} >= floor {int(floor)}; "
            f"predicted per-rank capacity {[int(x) for x in pred['p']]})"
        )
        lines.append(
            "floor check: predicted ctx of chosen vector "
            f"{int(pred['ctx'])} >= {int(floor)} "
            f"({100 - loose:g}% of VRAM-auto {int(base_pred['ctx'])}) -- OK"
        )
        lines.append(
            f"PIN HINT: skip probe+optimizer on later boots with "
            f"--rank-tp-ratio auto --rank-mlp-ratio "
            f"{','.join(map(str, chosen))}"
        )
    else:
        lines.append(
            "CHOSEN: keep plain VRAM-auto split (no MLP vector override)."
        )

    # Ratio-weighted vocab sharding hint (--rank-vocab-ratio, M20 BEIFANG 2):
    # a separate opt-in flag (never applied here); the membw-weighted vocab
    # split balances the per-rank lm_head read time (~+4% MTP decode class).
    if server_args.rank_vocab_ratio is None:
        vocab_vec = vocab_ratio_from_membw(rank_scores_bw)
        lines.append(
            "VOCAB HINT: the membw-weighted vocab shard for embed/lm_head "
            f"is --rank-vocab-ratio {','.join(map(str, vocab_vec))} "
            "(or 'auto'; separate opt-in flag, balances the per-rank "
            "lm_head read time -- helps MTP drafts and every sampling step)."
        )

    rec = _tp_drop_recommendation(server_args, profile, gpus, model)
    if rec:
        lines.append(rec)
    emit()


# ---------------------------------------------------------------------------
# Probe subprocess entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="sglang auto-performance probe")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.probe:
        out = args.out
        if out is None:
            gpus, driver = _nvml_gpu_inventory()
            out = profile_cache_path([g["uuid"] for g in gpus], driver)
        prof = run_probe(out)
        print(json.dumps(prof, indent=1))
    else:
        parser.error("nothing to do (pass --probe)")
