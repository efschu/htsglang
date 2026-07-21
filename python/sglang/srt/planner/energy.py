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
"""Energy + throughput MEASUREMENT harness (design #97 stage S2.5).

This is the MEASURED counterpart to ``roofline.py``'s estimate. The roofline
guesses tok/s from nameplate specs; this module BOOTS the real server (CUDA
graphs ON — full-perf discipline, never eager for a measurement), drives a
small batch-bucket sweep, and reports:

  * prefill / decode throughput (tok/s) per batch-size bucket, and
  * per-token ENERGY (Joules/token) per bucket, prefill-window and
    decode-window separated, by INTEGRATING NVML board power
    (``nvmlDeviceGetPowerUsage``) over the wall-clock of each phase, summed
    across ALL GPUs (whole-rig energy per token).

It produces a ``MeasurementResult`` and, crucially, ``ResultEntry`` rows with
``provenance="measured"`` that PASS the results_store ingest guard (which
rejects any unmeasured perf). The offline planner/dashboard then reads those
rows and prefers the MEASURED number over the roofline estimate.

WORKLOAD SPLIT (code vs prose)
------------------------------
Each bucket is measured under TWO fixed workloads — a CODE prompt+generation
and a PROSE prompt+generation, both temperature 0. For BASE decode (no MTP),
the per-token forward cost is identical regardless of content, so code and
prose decode tok/s / J-per-token land close here — we measure and report both
anyway because (a) prefill/gen lengths differ per workload, and (b) it sets up
the meaningful code-vs-prose divergence for a future MTP-on run (speculative
decoding accepts code drafts more often -> faster). Each workload is stored as
its own labelled ``ResultEntry`` (``workload="code"|"prose"``).

kWh per 1M tokens is NOT stored — it is a pure conversion of the measured
J/token (``kWh_per_1M = J_per_token / 3.6``) rendered next to it by the UI.

REUSABILITY
-----------
``run_measurement(MeasurementConfig)`` is the reusable entry point; the
``__main__`` block runs it on ONE validation config (Qwen3.6-27B-FP8, TP=3
uneven-DCP). Point it at any other serving config to fill that config's
measured perf/energy columns.

SHARED-BOX DISCIPLINE
---------------------
The harness owns exactly the server PID it spawns and tears down only that
process tree — never a broad ``pkill`` (other agents share this box).
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Workload",
    "MeasurementConfig",
    "BucketMeasurement",
    "MeasurementResult",
    "PowerSampler",
    "EnergyHarness",
    "run_measurement",
    "CODE_WORKLOAD",
    "PROSE_WORKLOAD",
    # #149 — power-state tagging + efficiency sweep + live monitoring
    "GpuPowerState",
    "read_gpu_power_states",
    "PowerLimitController",
    "SweepPoint",
    "SweepResult",
    "power_limit_sweep",
    "LiveSnapshot",
    "parse_prometheus_metrics",
    "compute_live_rates",
    "clamp_live_resolution_ms",
    "fetch_live_snapshot",
    "LIVE_MIN_RESOLUTION_MS",
    "LIVE_DEFAULT_RESOLUTION_MS",
    # #150 — configurable scenario builder + cache-flush safety gate
    "ScenarioConfig",
    "cache_flush_warning",
    "CACHE_FLUSH_ENDPOINTS",
]


# ---------------------------------------------------------------------------
# Workloads + config.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Workload:
    """A fixed (prompt, decode-length) pair, labelled by content class. The
    prompt drives PREFILL; ``decode_tokens`` forced tokens (ignore_eos) drive
    DECODE, so both phases have a known token count for the per-token math."""

    name: str  # "code" | "prose"
    prompt: str
    decode_tokens: int = 128


#: A representative code-completion prompt (structured, low-entropy — the case a
#: future MTP run drafts well).
CODE_WORKLOAD = Workload(
    name="code",
    prompt=(
        "Complete the following Python module. Write idiomatic, fully typed "
        "code with docstrings.\n\n"
        "```python\n"
        "import dataclasses\n"
        "from typing import Optional, List\n\n"
        "@dataclasses.dataclass\n"
        "class LRUCache:\n"
        '    """A fixed-capacity least-recently-used cache."""\n'
        "    capacity: int\n\n"
        "    def __post_init__(self) -> None:\n"
        "        self._store: dict = {}\n"
        "        self._order: List = []\n\n"
        "    def get(self, key):\n"
        "        # return the value for key, or None, and mark it most-recent\n"
    ),
    decode_tokens=128,
)

#: A representative natural-language essay prompt (higher-entropy prose).
PROSE_WORKLOAD = Workload(
    name="prose",
    prompt=(
        "Write a thoughtful, flowing essay of several paragraphs on how the "
        "history of cartography reflects humanity's changing relationship with "
        "the natural world. Discuss medieval mappae mundi, the Age of "
        "Exploration, the standardization of projections, and the modern era "
        "of satellite and digital mapping. Use vivid, varied language.\n\n"
    ),
    decode_tokens=128,
)


@dataclasses.dataclass
class MeasurementConfig:
    """Everything needed to boot + drive one serving config."""

    model_path: str
    served_model_name: str = "measured-model"
    tp_size: int = 1
    rank_gpu_id: Optional[List[int]] = None
    rank_tp_ratio: Optional[List[int]] = None
    rank_gpu_memory_mib: Optional[List[int]] = None
    kv_cache_dtype: str = "auto"
    context_length: int = 8192
    max_running_requests: int = 16
    quant_label: str = "fp8"  # for the results-store QuantDescriptor
    #: human label distinguishing this row from siblings (e.g. "no-MTP
    #: baseline" vs "MTP+adaptive"). Stored on every ResultEntry as config_label
    #: so the dashboard shows both rows side by side.
    label: str = "baseline"
    # -- speculative decoding (MTP / adaptive draft) ---------------------
    speculative_algorithm: Optional[str] = None  # e.g. "NEXTN"
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_adaptive: bool = False
    speculative_adaptive_config: Optional[str] = None  # "high-accept"
    speculative_adaptive_graph_memory: Optional[str] = None  # "auto" default
    #: batch buckets to sweep (concurrent request counts).
    buckets: Sequence[int] = (1, 4, 16)
    workloads: Sequence[Workload] = (CODE_WORKLOAD, PROSE_WORKLOAD)
    #: env overlaid on the server process (SGLANG_UNEVEN_DCP etc.).
    extra_env: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: extra raw launch flags appended verbatim.
    extra_flags: List[str] = dataclasses.field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 31000
    #: NVML indices to sample for whole-rig power (None -> every NVML device).
    power_gpu_indices: Optional[List[int]] = None
    power_sample_ms: float = 20.0
    boot_timeout_s: float = 900.0
    disable_cuda_graph: bool = False  # MUST stay False for a real measurement
    trust_remote_code: bool = True
    disable_custom_all_reduce: bool = True  # no P2P on this rig
    python_exe: str = sys.executable

    def launch_command(self) -> List[str]:
        c = [self.python_exe, "-m", "sglang.launch_server",
             "--model-path", self.model_path,
             "--served-model-name", self.served_model_name,
             "--tp-size", str(self.tp_size),
             "--kv-cache-dtype", self.kv_cache_dtype,
             "--context-length", str(self.context_length),
             "--max-running-requests", str(self.max_running_requests),
             "--host", self.host, "--port", str(self.port)]
        if self.rank_gpu_id is not None:
            c += ["--rank-gpu-id", ",".join(map(str, self.rank_gpu_id))]
        if self.rank_tp_ratio is not None:
            # string sentinel ("auto"/"auto-performance") passes through as-is;
            # an explicit tuple is comma-joined. Never str() a bare string (that
            # would emit "a,u,t,o").
            rtr = self.rank_tp_ratio
            c += ["--rank-tp-ratio",
                  rtr if isinstance(rtr, str) else ",".join(map(str, rtr))]
        if self.rank_gpu_memory_mib is not None:
            c += ["--rank-gpu-memory-mib",
                  ",".join(map(str, self.rank_gpu_memory_mib))]
        if self.trust_remote_code:
            c += ["--trust-remote-code"]
        if self.disable_custom_all_reduce:
            c += ["--disable-custom-all-reduce"]
        # Always expose Prometheus /metrics: the energy harness scrapes it for
        # live rates, and the dashboard landing page needs it for live decode/
        # prefill tok/s, MTP acceptance and per-tier cache-hit.
        c += ["--enable-metrics"]
        if self.disable_cuda_graph:
            c += ["--disable-cuda-graph"]
        if self.speculative_algorithm:
            c += ["--speculative-algorithm", self.speculative_algorithm]
        if self.speculative_num_steps is not None:
            c += ["--speculative-num-steps", str(self.speculative_num_steps)]
        if self.speculative_eagle_topk is not None:
            c += ["--speculative-eagle-topk", str(self.speculative_eagle_topk)]
        if self.speculative_num_draft_tokens is not None:
            c += ["--speculative-num-draft-tokens",
                  str(self.speculative_num_draft_tokens)]
        if self.speculative_adaptive:
            c += ["--speculative-adaptive"]
        if self.speculative_adaptive_config:
            c += ["--speculative-adaptive-config", self.speculative_adaptive_config]
        if self.speculative_adaptive_graph_memory:
            c += ["--speculative-adaptive-graph-memory",
                  self.speculative_adaptive_graph_memory]
        c += list(self.extra_flags)
        return c


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BucketMeasurement:
    """One (workload, bucket) point."""

    workload: str
    bucket: int
    prompt_tokens: int          # per request
    decode_tokens: int          # per request
    n_requests: int
    prefill_seconds: float      # wall-clock of the prompt-processing window
    decode_seconds: float       # wall-clock of the sustained-generation window
    prefill_tok_s: float
    decode_tok_s: float
    prefill_joules: float       # whole-rig SUM, integrated over prefill window
    decode_joules: float
    j_per_prefill_token: float  # whole-rig total per token
    j_per_decode_token: float
    avg_prefill_watts: float    # whole-rig sum
    avg_decode_watts: float
    #: PER-CARD breakdown (aligned to MeasurementResult.gpu_names_sampled). The
    #: whole-rig totals above are the sums of these. GPU (NVML) power only —
    #: excludes CPU/RAM/PSU-conversion losses (NOT wall-socket power).
    gpu_names: List[str] = dataclasses.field(default_factory=list)
    per_card_j_per_prefill_token: List[float] = dataclasses.field(default_factory=list)
    per_card_j_per_decode_token: List[float] = dataclasses.field(default_factory=list)
    per_card_avg_prefill_watts: List[float] = dataclasses.field(default_factory=list)
    per_card_avg_decode_watts: List[float] = dataclasses.field(default_factory=list)
    #: MTP/spec: avg verify forwards per request and the resulting accept length
    #: (output tokens per target forward = the measured MTP multiplier). None
    #: when speculative decoding is off.
    spec_verify_ct: Optional[float] = None
    accept_length: Optional[float] = None
    accept_rate: Optional[float] = None  # correct drafts / proposed (no bonus)

    def kwh_per_1m_prefill(self) -> float:
        return self.j_per_prefill_token / 3.6

    def kwh_per_1m_decode(self) -> float:
        return self.j_per_decode_token / 3.6


@dataclasses.dataclass
class MeasurementResult:
    config: MeasurementConfig
    hardware_cards: List[tuple]          # [(count, name, total_mib), ...]
    gpu_names_sampled: List[str]
    measurements: List[BucketMeasurement]
    idle_watts: float
    launch_flags: List[str]
    label: str = "baseline"
    #: Server-reported cumulative avg_spec_accept_length (authoritative, from
    #: /server_info) — output tokens accepted per target forward. None w/o spec.
    server_avg_accept_length: Optional[float] = None
    #: Adaptive controller's settled step (k) trajectory, parsed from the boot
    #: log: {k: times_switched_to_it}. Shows which depth the controller picks.
    adaptive_step_hist: Optional[Dict[int, int]] = None
    #: Per-NVML-card compute/weight share from the uneven-TP ratio (fraction of
    #: the model each physical card holds+computes), aligned to
    #: gpu_names_sampled. Lets the reader see whether a card draws more power
    #: because it does more work (expected on this heterogeneous rig) vs an
    #: efficiency difference. None for an even/single-card config.
    compute_share_by_card: Optional[List[float]] = None
    notes: List[str] = dataclasses.field(default_factory=list)

    # -- per-workload views --------------------------------------------------

    def by_workload(self, name: str) -> List[BucketMeasurement]:
        return [m for m in self.measurements if m.workload == name]

    def result_entries(self):
        """One measured ``ResultEntry`` per workload (labelled), populating the
        results_store's MEASURED perf/energy columns."""
        from sglang.srt.planner.results_store import QuantDescriptor, ResultEntry

        quant = QuantDescriptor.parse(self.config.quant_label)
        tp_cfg = "tp%d" % self.config.tp_size
        if self.config.rank_tp_ratio and len(set(self.config.rank_tp_ratio)) > 1:
            tp_cfg += " uneven " + ",".join(map(str, self.config.rank_tp_ratio))
        entries = []
        for wl in sorted({m.workload for m in self.measurements}):
            ms = self.by_workload(wl)
            prefill_tps = {m.bucket: m.prefill_tok_s for m in ms}
            decode_tps = {m.bucket: m.decode_tok_s for m in ms}
            j_pre = {m.bucket: m.j_per_prefill_token for m in ms}
            j_dec = {m.bucket: m.j_per_decode_token for m in ms}
            accept = {m.bucket: m.accept_length for m in ms
                      if m.accept_length is not None} or None
            gpu_names = ms[0].gpu_names if ms else self.gpu_names_sampled
            per_card_energy = {
                "gpu_names": list(gpu_names),
                "compute_share": (list(self.compute_share_by_card)
                                  if self.compute_share_by_card else None),
                "source": "GPU-measured (NVML), excludes CPU/PSU (not wall power)",
                "prefill_j_per_token_by_bucket":
                    {m.bucket: m.per_card_j_per_prefill_token for m in ms},
                "decode_j_per_token_by_bucket":
                    {m.bucket: m.per_card_j_per_decode_token for m in ms},
                "prefill_watts_by_bucket":
                    {m.bucket: m.per_card_avg_prefill_watts for m in ms},
                "decode_watts_by_bucket":
                    {m.bucket: m.per_card_avg_decode_watts for m in ms},
            }
            entries.append(ResultEntry(
                model=self.config.served_model_name,
                quant=quant,
                hardware_cards=list(self.hardware_cards),
                reproduce_flags=list(self.launch_flags),
                provenance="measured",
                tp_config=tp_cfg,
                workload=wl,
                prefill_tok_s_by_bucket=prefill_tps,
                decode_tok_s_by_bucket=decode_tps,
                peak_prefill_tok_s=max(prefill_tps.values()) if prefill_tps else None,
                peak_decode_tok_s=max(decode_tps.values()) if decode_tps else None,
                j_per_prefill_token_by_bucket=j_pre,
                j_per_decode_token_by_bucket=j_dec,
                per_card_energy=per_card_energy,
                config_label=self.label,
                spec_accept_length_by_bucket=accept,
                kv_cache_dtype=self.config.kv_cache_dtype,
            ))
        return entries


# ---------------------------------------------------------------------------
# NVML power sampler.
# ---------------------------------------------------------------------------


class PowerSampler:
    """Background thread sampling summed NVML board power (Watts) across the
    selected GPUs at a fixed cadence. ``integrate(t0, t1)`` returns the Joules
    consumed by the whole rig between two ``time.time()`` marks (trapezoidal
    integration over the samples in the window)."""

    def __init__(self, gpu_indices: Optional[Sequence[int]] = None,
                 sample_ms: float = 20.0):
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        idx = list(gpu_indices) if gpu_indices is not None else list(range(n))
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]
        self.gpu_names = [
            _nvml_name(pynvml.nvmlDeviceGetName(h)) for h in self._handles
        ]
        self._interval = sample_ms / 1000.0
        #: (t, [w_card0, w_card1, ...]) — PER-CARD watts kept, never only summed.
        self._samples: List[Tuple[float, List[float]]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_card_watts(self) -> List[float]:
        return [self._pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
                for h in self._handles]

    def _read_total_watts(self) -> float:
        return sum(self._read_card_watts())

    def _loop(self):
        while not self._stop.is_set():
            self._samples.append((time.time(), self._read_card_watts()))
            time.sleep(self._interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sample_now(self, seconds: float = 1.0) -> float:
        """Blocking mean-watts probe (used for the idle baseline)."""
        vals = []
        end = time.time() + seconds
        while time.time() < end:
            vals.append(self._read_total_watts())
            time.sleep(self._interval)
        return sum(vals) / len(vals) if vals else self._read_total_watts()

    def integrate(self, t0: float, t1: float) -> Tuple[float, float]:
        """(total_joules, total_mean_watts) over [t0, t1] — whole-rig sum."""
        pc_j, pc_w = self.integrate_per_card(t0, t1)
        return sum(pc_j), sum(pc_w)

    def integrate_per_card(
        self, t0: float, t1: float
    ) -> Tuple[List[float], List[float]]:
        """(per_card_joules, per_card_mean_watts) over [t0, t1] via trapezoidal
        integration — keeps the PER-CARD split (the 5090 vs the 3080s draw very
        different power). Whole-rig total is the caller's sum of these."""
        n = len(self._handles)
        pts = [(t, w) for (t, w) in self._samples if t0 <= t <= t1]
        if len(pts) < 2:
            w = list(pts[0][1]) if pts else self._read_card_watts()
            dt = max(t1 - t0, 1e-6)
            return [wi * dt for wi in w], list(w)
        joules = [0.0] * n
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            dt = tb - ta
            for i in range(n):
                joules[i] += 0.5 * (wa[i] + wb[i]) * dt
        span = pts[-1][0] - pts[0][0]
        mean_w = [j / span if span > 0 else pts[0][1][i]
                  for i, j in enumerate(joules)]
        return joules, mean_w

    def shutdown(self):
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


def _nvml_name(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


def _venv_cuda_lib_dirs(python_exe: str) -> List[str]:
    """The venv's bundled NVIDIA/torch shared-lib dirs (libcudart.so.13,
    libnvrtc.so.13, ...), so a RE-EXEC'd worker python finds them at interpreter
    startup (before torch's rpath is set up). Globs
    ``<venv>/lib/python*/site-packages/{nvidia/*/lib,torch/lib}``."""
    import glob

    base = os.path.dirname(os.path.dirname(os.path.abspath(python_exe)))  # venv/
    dirs = []
    for pat in ("lib/python*/site-packages/nvidia/*/lib",
                "lib/python*/site-packages/torch/lib"):
        dirs += sorted(glob.glob(os.path.join(base, pat)))
    # De-dup, keep existing order, only real dirs.
    seen, out = set(), []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# The harness.
# ---------------------------------------------------------------------------


class EnergyHarness:
    """Boots the server, drives the sweep, measures throughput + energy, tears
    the server down. Use as a context manager or call ``boot()`` / ``close()``."""

    def __init__(self, config: MeasurementConfig):
        self.cfg = config
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.host}:{config.port}"
        self.sampler: Optional[PowerSampler] = None
        self.notes: List[str] = []
        self._log_path = f"/tmp/energy_boot_{config.port}.log"
        self._log_file = None

    # -- lifecycle -----------------------------------------------------------

    def boot(self):
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "/spinning/wt-integration-r2/python")
        # --rank-gpu-id forces SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS, which
        # RE-EXECs each scheduler (and, under spec decoding, extra draft
        # workers) as a fresh python. A fresh interpreter loads libcudart before
        # torch sets up its rpath, so it dies with exit 127 ("error while
        # loading shared libraries: libcudart.so.13") unless the venv's bundled
        # CUDA libs are on LD_LIBRARY_PATH. Inject them so the re-exec'd workers
        # find them (the baseline sometimes forks and survives; the spec path
        # re-execs and does not).
        ld = _venv_cuda_lib_dirs(self.cfg.python_exe)
        if ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                ld + ([existing] if existing else []))
        env.update({k: str(v) for k, v in self.cfg.extra_env.items()})
        cmd = self.cfg.launch_command()
        self._log_file = open(self._log_path, "w")
        print(f"[energy] boot: {' '.join(cmd)}", flush=True)
        print(f"[energy] boot log -> {self._log_path}", flush=True)
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=self._log_file, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group -> clean, scoped teardown
        )
        self._wait_ready()

    def _wait_ready(self):
        deadline = time.time() + self.cfg.boot_timeout_s
        url = self.base_url + "/get_model_info"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self._log_tail()
                raise RuntimeError(
                    f"server exited during boot (rc={self.proc.returncode}). "
                    f"log tail:\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        print("[energy] server ready.", flush=True)
                        return
            except Exception:
                pass
            time.sleep(2.0)
        # Timed out -> diagnose (py-spy all ranks) before giving up.
        diag = self._diagnose_hang()
        raise TimeoutError(
            f"server not ready after {self.cfg.boot_timeout_s:.0f}s.\n"
            f"log tail:\n{self._log_tail()}\n\npy-spy:\n{diag}")

    def _log_tail(self, n: int = 60) -> str:
        try:
            with open(self._log_path) as f:
                return "".join(f.readlines()[-n:])
        except Exception:
            return "(no log)"

    def _diagnose_hang(self) -> str:
        """py-spy dump every python process in the server's process group —
        the uneven-DCP + FP8 + VL-vision + graph-capture combo can wedge in the
        vision-tower TP split or the decode-graph capture."""
        out = []
        try:
            pgid = os.getpgid(self.proc.pid)
            pids = subprocess.check_output(
                ["pgrep", "-g", str(pgid)], text=True).split()
        except Exception:
            pids = [str(self.proc.pid)]
        for pid in pids:
            try:
                dump = subprocess.check_output(
                    ["py-spy", "dump", "--pid", pid], text=True,
                    stderr=subprocess.STDOUT, timeout=30)
                out.append(f"--- pid {pid} ---\n{dump}")
            except Exception as e:
                out.append(f"--- pid {pid}: py-spy failed: {e} ---")
        return "\n".join(out) or "(no py-spy output)"

    def close(self):
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler.shutdown()
        if self.proc is not None and self.proc.poll() is None:
            # Scoped teardown: signal our own process group only.
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
                self._wait_exit(20)
            except Exception:
                pass
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._wait_exit(10)
        if self._log_file is not None:
            self._log_file.close()

    def _wait_exit(self, timeout: float):
        end = time.time() + timeout
        while time.time() < end and self.proc.poll() is None:
            time.sleep(0.5)

    def __enter__(self):
        self.boot()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- driving -------------------------------------------------------------

    def _generate_stream(self, prompt: str, max_new: int) -> dict:
        """One streaming /generate request. Returns per-request timing:
        t_send, t_first (TTFT wall), t_done, prompt_tokens, completion_tokens."""
        body = json.dumps({
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new,
                "ignore_eos": True,
            },
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/generate", data=body,
            headers={"Content-Type": "application/json"})
        t_send = time.time()
        t_first = None
        prompt_tokens = 0
        completion_tokens = 0
        spec_verify_ct = 0
        spec_correct = 0
        spec_proposed = 0
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                meta = obj.get("meta_info") or {}
                ct = meta.get("completion_tokens", 0)
                if t_first is None and ct and ct >= 1:
                    t_first = time.time()
                if ct:
                    completion_tokens = ct
                if meta.get("prompt_tokens"):
                    prompt_tokens = meta["prompt_tokens"]
                if meta.get("spec_verify_ct"):
                    spec_verify_ct = meta["spec_verify_ct"]
                if meta.get("spec_num_correct_drafts") is not None:
                    spec_correct = meta["spec_num_correct_drafts"]
                if meta.get("spec_num_proposed_drafts") is not None:
                    spec_proposed = meta["spec_num_proposed_drafts"]
        t_done = time.time()
        if t_first is None:
            t_first = t_done
        return {
            "t_send": t_send, "t_first": t_first, "t_done": t_done,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "spec_verify_ct": spec_verify_ct,
            "spec_correct": spec_correct, "spec_proposed": spec_proposed,
        }

    def _run_concurrent(self, prompt: str, max_new: int, n: int) -> List[dict]:
        results: List[dict] = [None] * n
        errors: List[Exception] = []

        def worker(i):
            try:
                results[i] = self._generate_stream(prompt, max_new)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results

    def _warmup(self, prompt: str):
        try:
            self._generate_stream(prompt, 8)
        except Exception as e:
            self.notes.append(f"warmup request failed: {e}")

    def measure(self) -> MeasurementResult:
        assert self.proc is not None, "boot() first"
        self.sampler = PowerSampler(
            self.cfg.power_gpu_indices, self.cfg.power_sample_ms)
        idle_watts = self.sampler.sample_now(2.0)
        self.sampler.start()

        # Warm up CUDA graphs / caches so the first measured bucket is not paying
        # a one-time capture/compile cost.
        self._warmup(self.cfg.workloads[0].prompt)

        measurements: List[BucketMeasurement] = []
        for wl in self.cfg.workloads:
            for bucket in self.cfg.buckets:
                bm = self._measure_point(wl, bucket)
                measurements.append(bm)
                acc = (f" accept={bm.accept_length:.2f}"
                       if bm.accept_length is not None else "")
                print(
                    f"[energy] {wl.name:5s} bs={bucket:<3d} "
                    f"prefill {bm.prefill_tok_s:8.1f} tok/s "
                    f"{bm.j_per_prefill_token:6.3f} J/tok | "
                    f"decode {bm.decode_tok_s:7.1f} tok/s "
                    f"{bm.j_per_decode_token:6.3f} J/tok "
                    f"({bm.avg_decode_watts:.0f} W){acc}", flush=True)

        self.sampler.stop()
        hardware_cards = _hardware_cards()
        share = _compute_share_by_nvml(self.cfg, self.sampler.gpu_names, self.notes)
        return MeasurementResult(
            config=self.cfg,
            hardware_cards=hardware_cards,
            gpu_names_sampled=self.sampler.gpu_names,
            measurements=measurements,
            idle_watts=idle_watts,
            launch_flags=self.cfg.launch_command()[3:],  # drop python -m module
            label=self.cfg.label,
            server_avg_accept_length=self._query_server_accept(),
            adaptive_step_hist=self._parse_adaptive_steps(),
            compute_share_by_card=share,
            notes=self.notes,
        )

    def _query_server_accept(self) -> Optional[float]:
        """Cumulative avg_spec_accept_length from /server_info (authoritative
        server-side counter). None when spec is off / unavailable."""
        try:
            with urllib.request.urlopen(self.base_url + "/server_info",
                                        timeout=10) as r:
                info = json.loads(r.read())
            states = info.get("internal_states") or []
            vals = [s.get("avg_spec_accept_length") for s in states
                    if s.get("avg_spec_accept_length") is not None]
            return sum(vals) / len(vals) if vals else None
        except Exception as e:
            self.notes.append(f"server accept-length query failed: {e}")
            return None

    def _parse_adaptive_steps(self) -> Optional[Dict[int, int]]:
        """Parse the boot log for the adaptive controller's step (k) picks:
        'Adaptive spec params updated: steps X -> Y (...)'. Returns {k: count of
        switches INTO k} so the report shows which depth it settles on."""
        if not self.cfg.speculative_adaptive:
            return None
        import re
        hist: Dict[int, int] = {}
        try:
            with open(self._log_path) as f:
                for line in f:
                    m = re.search(r"Adaptive spec params updated: steps \d+ -> "
                                  r"(\d+)", line)
                    if m:
                        k = int(m.group(1))
                        hist[k] = hist.get(k, 0) + 1
                    mi = re.search(r"AdaptiveSpeculativeParams initialized: "
                                   r"steps=(\d+)", line)
                    if mi:
                        k = int(mi.group(1))
                        hist.setdefault(k, 0)
        except Exception:
            pass
        return hist or None

    def _measure_point(self, wl: Workload, bucket: int) -> BucketMeasurement:
        # brief settle so the previous bucket's power tail does not bleed in.
        time.sleep(0.5)
        res = self._run_concurrent(wl.prompt, wl.decode_tokens, bucket)
        t_send = min(r["t_send"] for r in res)
        # Prefill window ends when the LAST request has produced its first token
        # (whole batch is out of prefill and into decode).
        t_prefill_end = max(r["t_first"] for r in res)
        t_done = max(r["t_done"] for r in res)

        prompt_tokens = int(round(
            sum(r["prompt_tokens"] for r in res) / len(res)))
        decode_tokens = int(round(
            sum(r["completion_tokens"] for r in res) / len(res)))
        n = len(res)

        # MTP/spec: accept length = output tokens per target-model verify
        # forward (= the measured multiplier); accept rate = correct drafts /
        # proposed drafts (strict, no bonus). None when spec is off.
        spec_verify_ct = accept_length = accept_rate = None
        total_verify = sum(r.get("spec_verify_ct") or 0 for r in res)
        if total_verify > 0:
            total_out = sum(r["completion_tokens"] for r in res)
            total_correct = sum(r.get("spec_correct") or 0 for r in res)
            total_proposed = sum(r.get("spec_proposed") or 0 for r in res)
            spec_verify_ct = total_verify / n
            accept_length = total_out / total_verify
            accept_rate = (total_correct / total_proposed
                           if total_proposed > 0 else None)

        prefill_time = max(t_prefill_end - t_send, 1e-6)
        decode_time = max(t_done - t_prefill_end, 1e-6)
        total_prompt = prompt_tokens * n
        total_decode = decode_tokens * n

        pc_pre_j, pc_pre_w = self.sampler.integrate_per_card(t_send, t_prefill_end)
        pc_dec_j, pc_dec_w = self.sampler.integrate_per_card(t_prefill_end, t_done)
        prefill_j, decode_j = sum(pc_pre_j), sum(pc_dec_j)

        def _per_tok(joules_list, toks):
            return [j / toks if toks else 0.0 for j in joules_list]

        return BucketMeasurement(
            workload=wl.name, bucket=bucket,
            prompt_tokens=prompt_tokens, decode_tokens=decode_tokens,
            n_requests=n,
            prefill_seconds=prefill_time, decode_seconds=decode_time,
            prefill_tok_s=total_prompt / prefill_time,
            decode_tok_s=total_decode / decode_time,
            prefill_joules=prefill_j, decode_joules=decode_j,
            j_per_prefill_token=prefill_j / total_prompt if total_prompt else 0.0,
            j_per_decode_token=decode_j / total_decode if total_decode else 0.0,
            avg_prefill_watts=sum(pc_pre_w), avg_decode_watts=sum(pc_dec_w),
            gpu_names=list(self.sampler.gpu_names),
            per_card_j_per_prefill_token=_per_tok(pc_pre_j, total_prompt),
            per_card_j_per_decode_token=_per_tok(pc_dec_j, total_decode),
            per_card_avg_prefill_watts=list(pc_pre_w),
            per_card_avg_decode_watts=list(pc_dec_w),
            spec_verify_ct=spec_verify_ct, accept_length=accept_length,
            accept_rate=accept_rate,
        )


def _norm_uuid(u) -> str:
    s = u.decode() if isinstance(u, bytes) else str(u)
    return "".join(ch for ch in s.lower() if ch in "0123456789abcdef")


def _torch_to_nvml_index() -> Dict[int, int]:
    """{torch_cuda_index: nvml_index} matched by GPU UUID. The runtime places
    ranks in torch.cuda order (5090 first here, FASTEST_FIRST) while NVML/
    nvidia-smi enumerate by PCI bus (5090 at index 1 on this rig) — the
    device-order trap. UUID matching is order-independent. Identity fallback."""
    import pynvml

    pynvml.nvmlInit()
    nvml_by_uuid = {}
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        nvml_by_uuid[_norm_uuid(pynvml.nvmlDeviceGetUUID(h))] = i
    mapping: Dict[int, int] = {}
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            uu = getattr(torch.cuda.get_device_properties(i), "uuid", None)
            mapping[i] = nvml_by_uuid.get(_norm_uuid(uu), i) if uu else i
    except Exception:
        pass
    return mapping


def _compute_share_by_nvml(cfg, gpu_names, notes) -> Optional[List[float]]:
    """Per-NVML-card compute/weight share (fraction of the model) from the
    uneven-TP ratio, aligned to the NVML-sampled card order. Each rank's ratio
    weight is attributed to the physical card that rank runs on (torch->NVML
    by UUID). Even/single-card configs return None."""
    ratio = cfg.rank_tp_ratio
    ids = cfg.rank_gpu_id
    if not ratio or not ids or len(set(ratio)) <= 1:
        return None
    total = float(sum(ratio))
    try:
        t2n = _torch_to_nvml_index()
    except Exception:
        t2n = {}
    per = [0.0] * len(gpu_names)
    for r, torch_dev in enumerate(ids):
        nvml_idx = t2n.get(torch_dev, torch_dev)
        if 0 <= nvml_idx < len(per):
            per[nvml_idx] += ratio[r] / total
        else:
            notes.append(
                f"compute-share: rank {r} torch dev {torch_dev} -> NVML "
                f"{nvml_idx} out of range; share unattributed.")
    return per


def _hardware_cards() -> List[tuple]:
    """Anonymous [(count, name, total_mib), ...] hardware class from NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        counts: Dict[Tuple[str, int], int] = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = _nvml_name(pynvml.nvmlDeviceGetName(h))
            total_mib = int(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20)
            key = (name, total_mib)
            counts[key] = counts.get(key, 0) + 1
        pynvml.nvmlShutdown()
        return [(c, n, m) for (n, m), c in counts.items()]
    except Exception:
        return []


# ===========================================================================
# #149 — power-state TAGGING, power-limit efficiency SWEEP, LIVE monitoring.
# ===========================================================================
#
# Every NVML-touching class/function here takes an injectable ``nvml`` object
# (defaulting to the real ``pynvml`` module) so the pure logic — the tagging
# schema, the ALWAYS-restore-limit guarantee, the physical-impossibility of a
# limit outside the card's constraints, the live delta-rate math — is unit
# testable on a box with no GPU (see test_energy.py).


def _import_pynvml():
    import pynvml

    pynvml.nvmlInit()
    return pynvml


def _resolve_handles(nvml, gpu_indices: Optional[Sequence[int]]):
    n = nvml.nvmlDeviceGetCount()
    idx = list(gpu_indices) if gpu_indices is not None else list(range(n))
    return idx, [nvml.nvmlDeviceGetHandleByIndex(i) for i in idx]


@dataclasses.dataclass
class GpuPowerState:
    """A point-in-time POWER-STATE TAG for one physical card. Everything the
    reader needs to judge whether two efficiency numbers are comparable: the
    actual draw, the *limit context* it was drawn under, the clocks, the temp,
    and whether the card is running stock / over- / under-clocked-or-limited.
    Efficiency (J/token) is only comparable across cards/runs at the SAME
    power-limit context — this tag makes that context explicit."""

    nvml_index: int
    name: str
    power_watts: float            # measured actual board draw right now
    power_limit_w: float          # current ENFORCED management limit
    default_limit_w: float        # factory/stock TDP
    min_limit_w: float
    max_limit_w: float
    limit_pct_of_default: float   # current limit / stock TDP (the sweep x-axis)
    sm_clock_mhz: int
    mem_clock_mhz: int
    temperature_c: int
    #: "stock" | "power-limited" (limit < default) | "power-raised"
    #: (limit > default) | "clock-OC" | "clock-UV" (graphics-clock offset, when
    #: NVML exposes it). A non-"stock" tag means this card's efficiency number
    #: is NOT directly comparable to a stock card without normalizing the axis.
    oc_uv: str = "stock"
    clock_offset_mhz: Optional[int] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _classify_oc_uv(limit_w, default_w, offset_mhz) -> str:
    if offset_mhz is not None and offset_mhz != 0:
        return "clock-OC" if offset_mhz > 0 else "clock-UV"
    if default_w > 0:
        if limit_w < default_w - 1.0:
            return "power-limited"
        if limit_w > default_w + 1.0:
            return "power-raised"
    return "stock"


def read_gpu_power_states(
    gpu_indices: Optional[Sequence[int]] = None, nvml=None
) -> List[GpuPowerState]:
    """Tag every selected card's current power state (see GpuPowerState)."""
    own = nvml is None
    nvml = nvml or _import_pynvml()
    try:
        idx, handles = _resolve_handles(nvml, gpu_indices)
        out: List[GpuPowerState] = []
        for i, h in zip(idx, handles):
            limit = nvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            default = nvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0
            try:
                mn, mx = nvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
                mn, mx = mn / 1000.0, mx / 1000.0
            except Exception:
                mn, mx = default, default
            offset = None
            try:
                offset = int(nvml.nvmlDeviceGetGpcClkVfOffset(h))
            except Exception:
                offset = None
            out.append(GpuPowerState(
                nvml_index=i,
                name=_nvml_name(nvml.nvmlDeviceGetName(h)),
                power_watts=nvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                power_limit_w=limit,
                default_limit_w=default,
                min_limit_w=mn,
                max_limit_w=mx,
                limit_pct_of_default=(limit / default if default else 1.0),
                sm_clock_mhz=_nvml_clock(nvml, h, "SM"),
                mem_clock_mhz=_nvml_clock(nvml, h, "MEM"),
                temperature_c=_nvml_temp(nvml, h),
                oc_uv=_classify_oc_uv(limit, default, offset),
                clock_offset_mhz=offset,
            ))
        return out
    finally:
        if own:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def _nvml_clock(nvml, h, which) -> int:
    try:
        clk = getattr(nvml, "NVML_CLOCK_SM", 1) if which == "SM" else \
            getattr(nvml, "NVML_CLOCK_MEM", 2)
        return int(nvml.nvmlDeviceGetClockInfo(h, clk))
    except Exception:
        return 0


def _nvml_temp(nvml, h) -> int:
    try:
        sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
        return int(nvml.nvmlDeviceGetTemperature(h, sensor))
    except Exception:
        return 0


class PowerLimitController:
    """Sets per-card NVML power-management limits and — via the context-manager
    protocol AND explicit ``restore()`` — ALWAYS restores the original limits,
    on clean exit and on error alike. The efficiency sweep never leaves a card
    clamped: even if a measurement raises mid-sweep, ``__exit__`` restores.

    A requested limit is clamped to the card's ``[min, max]`` constraints and
    the *physically applied* value is returned, so the caller records the real
    limit each point was measured at (not the requested one)."""

    def __init__(self, gpu_indices: Optional[Sequence[int]] = None, nvml=None):
        self._own = nvml is None
        self._nvml = nvml or _import_pynvml()
        self._idx, self._handles = _resolve_handles(self._nvml, gpu_indices)
        # Snapshot the ORIGINAL enforced limits up front (mW) — the restore set.
        self._original_mw: List[int] = [
            int(self._nvml.nvmlDeviceGetPowerManagementLimit(h))
            for h in self._handles
        ]
        self._default_mw: List[int] = [
            int(self._nvml.nvmlDeviceGetPowerManagementDefaultLimit(h))
            for h in self._handles
        ]
        self._constraints: List[Tuple[int, int]] = []
        for h in self._handles:
            try:
                mn, mx = self._nvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
            except Exception:
                d = int(self._nvml.nvmlDeviceGetPowerManagementDefaultLimit(h))
                mn, mx = d, d
            self._constraints.append((int(mn), int(mx)))
        self._restored = False

    @property
    def default_watts(self) -> List[float]:
        return [d / 1000.0 for d in self._default_mw]

    def _clamp_mw(self, card: int, want_mw: int) -> int:
        mn, mx = self._constraints[card]
        return max(mn, min(mx, int(want_mw)))

    def set_limit_pct(self, pct: float) -> List[float]:
        """Set every card to *pct* of its OWN stock TDP; return applied W."""
        applied = []
        for card, h in enumerate(self._handles):
            want = self._clamp_mw(card, round(self._default_mw[card] * pct))
            self._nvml.nvmlDeviceSetPowerManagementLimit(h, want)
            applied.append(want / 1000.0)
        return applied

    def set_limit_watts(self, watts: float) -> List[float]:
        """Set every card to *watts* absolute (clamped); return applied W."""
        applied = []
        for card, h in enumerate(self._handles):
            want = self._clamp_mw(card, round(watts * 1000.0))
            self._nvml.nvmlDeviceSetPowerManagementLimit(h, want)
            applied.append(want / 1000.0)
        return applied

    def restore(self) -> None:
        """Restore every card to the limit it had when this controller was
        constructed. Idempotent; best-effort per card so one failure does not
        strand the others."""
        if self._restored:
            return
        for h, mw in zip(self._handles, self._original_mw):
            try:
                self._nvml.nvmlDeviceSetPowerManagementLimit(h, mw)
            except Exception:
                pass
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        if self._own:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


@dataclasses.dataclass
class SweepPoint:
    """One power-limit rung of the efficiency sweep."""

    limit_pct: float                 # requested fraction of stock TDP
    applied_watts: List[float]       # actual per-card limit applied (clamped)
    total_applied_watts: float       # whole-rig limit sum (abs-W x-axis)
    decode_tok_s: float
    j_per_decode_token: float
    avg_decode_watts: float
    prefill_tok_s: float = 0.0
    j_per_prefill_token: float = 0.0


@dataclasses.dataclass
class SweepResult:
    """A full 60-100% power-limit efficiency sweep, plus the three comparison
    baselines the UI can plot the x-axis against."""

    points: List[SweepPoint]
    stock_total_watts: float         # sum of per-card stock TDP (100% baseline)
    gpu_names: List[str]

    def x_axis(self, mode: str = "pct_tdp") -> List[float]:
        """X values for each point. ``mode``:
          * ``"pct_tdp"`` — limit as a fraction of stock TDP (0-1),
          * ``"abs_w"``   — absolute whole-rig limit in Watts,
          * ``"stock"``   — same as pct_tdp but 1.0 marks the stock baseline."""
        if mode in ("pct_tdp", "stock"):
            return [p.limit_pct for p in self.points]
        if mode == "abs_w":
            return [p.total_applied_watts for p in self.points]
        raise ValueError(f"unknown x-axis mode: {mode!r}")

    def most_efficient(self) -> Optional[SweepPoint]:
        """The rung with the lowest decode J/token (best energy efficiency)."""
        valid = [p for p in self.points if p.j_per_decode_token > 0]
        return min(valid, key=lambda p: p.j_per_decode_token) if valid else None

    def to_json(self) -> dict:
        return {
            "gpu_names": list(self.gpu_names),
            "stock_total_watts": self.stock_total_watts,
            "points": [dataclasses.asdict(p) for p in self.points],
        }


def power_limit_sweep(
    measure_fn,
    pcts: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
    gpu_indices: Optional[Sequence[int]] = None,
    nvml=None,
    settle_s: float = 2.0,
    sleep_fn=time.sleep,
) -> SweepResult:
    """Set each power-limit rung in *pcts* (fraction of stock TDP), run
    *measure_fn()* -> ``BucketMeasurement``-like object (needs ``decode_tok_s``,
    ``j_per_decode_token``, ``avg_decode_watts``; prefill fields optional), and
    record an efficiency point. The original limits are ALWAYS restored via the
    ``PowerLimitController`` context manager — clean exit AND any exception."""
    ctrl = PowerLimitController(gpu_indices, nvml=nvml)
    names = [_nvml_name(ctrl._nvml.nvmlDeviceGetName(h)) for h in ctrl._handles]
    stock_total = sum(ctrl.default_watts)
    points: List[SweepPoint] = []
    with ctrl:
        for pct in pcts:
            applied = ctrl.set_limit_pct(pct)
            if settle_s:
                sleep_fn(settle_s)  # let clocks settle at the new cap
            m = measure_fn()
            points.append(SweepPoint(
                limit_pct=pct,
                applied_watts=applied,
                total_applied_watts=sum(applied),
                decode_tok_s=getattr(m, "decode_tok_s", 0.0),
                j_per_decode_token=getattr(m, "j_per_decode_token", 0.0),
                avg_decode_watts=getattr(m, "avg_decode_watts", 0.0),
                prefill_tok_s=getattr(m, "prefill_tok_s", 0.0),
                j_per_prefill_token=getattr(m, "j_per_prefill_token", 0.0),
            ))
    return SweepResult(points=points, stock_total_watts=stock_total,
                       gpu_names=names)


# ---------------------------------------------------------------------------
# LIVE monitoring: poll a RUNNING server's Prometheus /metrics, compute
# client-side delta rates. Token counters (sglang:prompt_tokens_total /
# sglang:generation_tokens_total) are ALREADY Prometheus Counters, so
# prefill/s + decode/s are pure client-side deltas of the raw counters — far
# finer-grained than scheduler.last_gen_throughput (a ~1s server-side average).
# The spec signals (accept rate, active adaptive-k, and the newly-exposed
# ema_accept_len) are gauges read straight off the same scrape.
# ---------------------------------------------------------------------------

#: floor resolution — polling faster than this is pointless (scrape + parse
#: latency dominates) and hammers the server.
LIVE_MIN_RESOLUTION_MS = 30.0
LIVE_DEFAULT_RESOLUTION_MS = 250.0


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus text-exposition into ``{metric_name: summed_value}``,
    summing across all label sets of the same metric (e.g. per-worker token
    counters collapse to a rig-wide total). Comments (# HELP/# TYPE) skipped."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        if brace >= 0:
            name = line[:brace]
            rest = line[line.find("}", brace) + 1:].strip()
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, rest = parts[0], parts[1]
        val = rest.split()[0] if rest else ""
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


@dataclasses.dataclass
class LiveSnapshot:
    """One /metrics scrape, timestamped, with the fields the live widget needs."""

    t: float
    prompt_tokens_total: float
    generation_tokens_total: float
    spec_accept_rate: float
    spec_num_steps: float          # current adaptive-k
    spec_ema_accept_len: float
    gen_throughput: float          # server-side ~1s avg (coarse; for reference)

    @classmethod
    def from_metrics(cls, metrics: Dict[str, float], t: Optional[float] = None):
        g = metrics.get
        return cls(
            t=t if t is not None else time.time(),
            prompt_tokens_total=g("sglang:prompt_tokens_total", 0.0),
            generation_tokens_total=g("sglang:generation_tokens_total", 0.0),
            spec_accept_rate=g("sglang:spec_accept_rate", 0.0),
            spec_num_steps=g("sglang:spec_num_steps", 0.0),
            spec_ema_accept_len=g("sglang:spec_ema_accept_len", 0.0),
            gen_throughput=g("sglang:gen_throughput", 0.0),
        )


def compute_live_rates(prev: LiveSnapshot, cur: LiveSnapshot) -> Dict[str, float]:
    """Client-side delta rates between two scrapes. ``prefill_tok_s`` and
    ``decode_tok_s`` come from the RAW counter deltas over the wall-clock gap
    (not the server's coarse average). Spec gauges are point-in-time reads.
    A non-positive dt (clock jitter / duplicate scrape) yields zero rates
    rather than a divide-by-zero blow-up."""
    dt = cur.t - prev.t
    if dt <= 0:
        prefill = decode = 0.0
    else:
        prefill = max(0.0, cur.prompt_tokens_total - prev.prompt_tokens_total) / dt
        decode = max(0.0, cur.generation_tokens_total - prev.generation_tokens_total) / dt
    return {
        "dt": dt,
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "accept_rate": cur.spec_accept_rate,
        "adaptive_k": cur.spec_num_steps,
        "ema_accept_len": cur.spec_ema_accept_len,
        "gen_throughput_server": cur.gen_throughput,
    }


def clamp_live_resolution_ms(ms: float) -> float:
    """Clamp a requested widget resolution to the ~30ms floor."""
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return LIVE_DEFAULT_RESOLUTION_MS
    return max(LIVE_MIN_RESOLUTION_MS, ms)


def fetch_live_snapshot(base_url: str, timeout: float = 5.0) -> LiveSnapshot:
    """Scrape ``{base_url}/metrics`` once and return a LiveSnapshot. Used by the
    'measure against a running server' live mode — no boot, no teardown."""
    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    return LiveSnapshot.from_metrics(parse_prometheus_metrics(text))


# ===========================================================================
# #150 — configurable SCENARIO builder + cache-flush safety gate.
# ===========================================================================
#
# CORRECTED METHODOLOGY (the user's point): PREFILL cost is content-independent
# (prompt tokens go through the same dense forward regardless of whether they
# are code or prose), so prefill is measured ONCE — no code/prose split. DECODE
# is what diverges (MTP accepts code drafts more often than prose), so decode is
# measured per behavior. COLD prefill flushes the caches + uses a FRESH random
# prompt each run so no prefix-cache hit shortcuts the measured prefill.


CODE_BEHAVIOR = "code"
PROSE_BEHAVIOR = "prose"


@dataclasses.dataclass
class ScenarioConfig:
    """A user-configurable measurement scenario (#150). Fully declarative — the
    UI builds one of these and ``expand()`` turns it into the concrete list of
    phase/behavior runs the harness executes."""

    #: linear scale on the token lengths (0.1 = quick smoke, 1.0 = short-test
    #: default of 10k prefill / 1k decode, >1 = bigger).
    scale: float = 1.0
    #: "prefill" (prefill-only), "decode" (decode-only), "both".
    phases: str = "both"
    #: concurrent request count for this scenario (1 = single, N = batched).
    concurrency: int = 1
    #: which decode behaviors to split on. Prefill ignores this (content-
    #: independent). Default measures both code and prose decode.
    behaviors: Sequence[str] = (CODE_BEHAVIOR, PROSE_BEHAVIOR)
    #: base short-test lengths (scaled by ``scale``).
    prefill_tokens: int = 10_000
    decode_tokens: int = 1_000
    #: approximate wall-clock budget per phase run (seconds); the short-test
    #: default is ~10s.
    duration_s: float = 10.0
    #: MULTITURN: model a conversation whose context GROWS each turn and is
    #: re-prefilled. ``turns`` turns, each adding ``turn_growth_tokens`` of
    #: context. single-shot when False.
    multiturn: bool = False
    turns: int = 1
    turn_growth_tokens: int = 2_000
    #: COLD prefill: flush KV + hicache and use a fresh randomized prompt before
    #: each prefill run so no prefix-cache hit shortcuts the measured prefill.
    cold_prefill: bool = True

    def __post_init__(self):
        if self.phases not in ("prefill", "decode", "both"):
            raise ValueError(
                f"phases must be prefill|decode|both, got {self.phases!r}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        bad = [b for b in self.behaviors if b not in (CODE_BEHAVIOR, PROSE_BEHAVIOR)]
        if bad:
            raise ValueError(f"unknown behaviors: {bad}")
        if self.multiturn and self.turns < 1:
            raise ValueError(f"multiturn turns must be >= 1, got {self.turns}")

    def scaled_prefill_tokens(self) -> int:
        return max(1, int(round(self.prefill_tokens * self.scale)))

    def scaled_decode_tokens(self) -> int:
        return max(1, int(round(self.decode_tokens * self.scale)))

    def expand(self) -> List[dict]:
        """Concrete ordered list of run units. Each unit is a dict:
          {phase: "prefill"|"decode", behavior, prompt_tokens, decode_tokens,
           concurrency, cold, turn (multiturn only)}.
        PREFILL emits ONE unit (behavior "any" — content-independent); DECODE
        emits one unit per behavior. Multiturn expands into ``turns`` units with
        growing prefill context, each preceded by a re-prefill."""
        prefill_tok = self.scaled_prefill_tokens()
        decode_tok = self.scaled_decode_tokens()
        units: List[dict] = []

        def _turns_iter():
            if not self.multiturn:
                yield 0, prefill_tok
                return
            ctx = prefill_tok
            for turn in range(self.turns):
                yield turn, ctx
                ctx += self.turn_growth_tokens  # context GROWS -> re-prefill

        if self.phases in ("prefill", "both"):
            for turn, ctx in _turns_iter():
                units.append({
                    "phase": "prefill",
                    "behavior": "any",          # content-independent
                    "prompt_tokens": ctx,
                    "decode_tokens": 0,
                    "concurrency": self.concurrency,
                    "cold": self.cold_prefill,
                    "turn": turn,
                })
        if self.phases in ("decode", "both"):
            for behavior in self.behaviors:
                for turn, ctx in _turns_iter():
                    units.append({
                        "phase": "decode",
                        "behavior": behavior,
                        "prompt_tokens": ctx,
                        "decode_tokens": decode_tok,
                        "concurrency": self.concurrency,
                        # decode re-prefills its growing context each turn but
                        # the measured window is decode; only prefill units are
                        # marked cold (they own the flush).
                        "cold": self.cold_prefill and self.phases == "decode",
                        "turn": turn,
                    })
        return units


#: The endpoints a cold-prefill / benchmark flush hits on the target server.
CACHE_FLUSH_ENDPOINTS = ("/flush_cache", "/clear_hicache_storage_backend")


def cache_flush_warning(
    will_flush: bool, target_running_server: bool
) -> dict:
    """Decide whether the UI must warn before a cache flush, and whether the
    confirmation is MANDATORY (#150, the user's latest point).

    A flush wipes KV + hierarchical cache — any context a user wanted to keep
    is lost. So:
      * targeting a RUNNING (production) server -> confirmation is MANDATORY
        (a blocking confirm the user must accept),
      * a FRESH measurement server we spawned -> informative warning only
        (nothing precious to lose), non-blocking.
      * not flushing at all -> no warning.

    Returns ``{"warn", "mandatory", "message", "endpoints"}``; the UI enforces
    the gate (blocks the action until confirmed when ``mandatory``)."""
    if not will_flush:
        return {"warn": False, "mandatory": False, "message": "",
                "endpoints": []}
    if target_running_server:
        msg = ("Cache will be cleared — context you may want to keep is lost. "
               "This targets a RUNNING server. Continue?")
        return {"warn": True, "mandatory": True, "message": msg,
                "endpoints": list(CACHE_FLUSH_ENDPOINTS)}
    msg = ("Cache will be cleared on this fresh measurement server before the "
           "cold-prefill run (nothing in production is affected).")
    return {"warn": True, "mandatory": False, "message": msg,
            "endpoints": list(CACHE_FLUSH_ENDPOINTS)}


def run_measurement(config: MeasurementConfig) -> MeasurementResult:
    """Boot -> measure -> teardown. The reusable one-call entry point."""
    harness = EnergyHarness(config)
    try:
        harness.boot()
        return harness.measure()
    finally:
        harness.close()


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------


def _work_per_watt_rel(
    shares: Sequence[Optional[float]], j_per_token: Sequence[float]
) -> List[Optional[float]]:
    """RELATIVE work-per-watt efficiency per card (the coordinator's column).

    Under pure TP every card contributes to every token at similar wattage, so
    raw J/token hides which card is the efficient workhorse. Normalizing by the
    WORK each card does — its compute-share from ``--rank-tp-ratio`` — exposes
    it:

        efficiency_i = compute_share_i / j_per_token_i     (work per energy)

    Within one (workload, bucket) row every card shares the same decode window
    and total-token count, so ``j_per_token_i`` is proportional to that card's
    average watts; hence ``share / j_per_token`` and ``share / watts`` give the
    IDENTICAL relative ordering (the coordinator's "or equivalently J/tok ÷
    share"). The result is scaled so the LEAST-efficient card reads ``1.0x``
    and more-efficient cards read higher (e.g. the 5090 ~2.49x in prefill,
    ~1.78x in decode vs the 3080s on this rig). Cards without a known share
    (even/single-card configs) return ``None``."""
    eff: List[Optional[float]] = []
    for i, jt in enumerate(j_per_token):
        sh = shares[i] if i < len(shares) else None
        if sh is None or jt <= 0:
            eff.append(None)
        else:
            eff.append(sh / jt)
    valid = [e for e in eff if e is not None and e > 0]
    if not valid:
        return [None] * len(j_per_token)
    base = min(valid)  # least-efficient card -> 1.0x
    return [(e / base if e is not None and base > 0 else None) for e in eff]


def summarize(result: MeasurementResult, price_ct_per_kwh: float = 30.0) -> str:
    """Full results view. PREFILL and DECODE are shown as SEPARATE, self-
    contained blocks (each with tok/s + J/tok + kWh/1M + ct/1M) — never summed
    into one number, because prefill (compute, amortized over the prompt) and
    decode (memory-bound, per output token) have very different per-token cost.
    ``price_ct_per_kwh`` converts kWh/1M -> ct/1M."""
    def ct1m(j):  # J/token -> ct per 1M tokens
        return (j / 3.6) * price_ct_per_kwh

    lines = []
    lines.append("=" * 88)
    lines.append(f"MEASURED energy + throughput — [{result.label}] "
                 f"(CUDA graphs ON, temp 0)")
    lines.append(f"  rig: {', '.join(f'{c}x {n}' for c, n, m in result.hardware_cards)}")
    lines.append(f"  idle whole-rig: {result.idle_watts:.0f} W  |  price: "
                 f"{price_ct_per_kwh:.0f} ct/kWh")
    lines.append("  energy = GPU (NVML) board power only, summed across cards — "
                 "EXCLUDES CPU/RAM/PSU losses (NOT wall-socket power).")
    if result.compute_share_by_card:
        shares = ", ".join(
            f"{n} {s*100:.0f}%" for n, s in
            zip(result.gpu_names_sampled, result.compute_share_by_card))
        lines.append(f"  uneven-TP compute share (from ratio): {shares}")
    if result.server_avg_accept_length is not None:
        lines.append(f"  MTP avg accept length (server /server_info): "
                     f"{result.server_avg_accept_length:.2f} output tok / target "
                     "forward")
    if result.adaptive_step_hist:
        hist = ", ".join(f"k={k} (x{c})" for k, c in
                         sorted(result.adaptive_step_hist.items()))
        lines.append(f"  adaptive draft depth picks (ceiling k=5): {hist}")
    lines.append("=" * 88)

    def phase_block(title, tok_s, jtok, accept=False):
        b = [title]
        hdr = (f"{'workload':8s} {'bs':>3s} {'tok/s':>10s} {'J/tok':>9s} "
               f"{'kWh/1M':>9s} {'ct/1M':>9s}")
        if accept:
            hdr += f" {'acc-len':>7s} {'acc-rate':>8s}"
        b.append(hdr)
        b.append("-" * len(hdr))
        for m in result.measurements:
            ts, jt = tok_s(m), jtok(m)
            row = (f"{m.workload:8s} {m.bucket:3d} {ts:10.1f} {jt:9.3f} "
                   f"{jt/3.6:9.3f} {ct1m(jt):9.3f}")
            if accept:
                row += (f" {m.accept_length:7.2f}" if m.accept_length is not None
                        else f" {'—':>7s}")
                row += (f" {m.accept_rate*100:7.1f}%" if m.accept_rate is not None
                        else f" {'—':>8s}")
            b.append(row)
        return b

    # -- TWO separate self-contained blocks: PREFILL and DECODE --------------
    lines += phase_block(
        "PREFILL — its own operation (compute-bound, amortized over the prompt):",
        lambda m: m.prefill_tok_s, lambda m: m.j_per_prefill_token)
    lines.append("")
    lines += phase_block(
        "DECODE — its own operation (memory-bound, per OUTPUT token):",
        lambda m: m.decode_tok_s, lambda m: m.j_per_decode_token, accept=True)

    # -- PER-CARD: also separated into prefill + decode ----------------------
    names = result.gpu_names_sampled
    shares = result.compute_share_by_card or [None] * len(names)

    def per_card_block(title, jtok_list, watt_list):
        b = ["", title]
        b.append("  raw J/tok is the honest per-TOTAL-token number, but it HIDES "
                 "efficiency: under TP every card")
        b.append("  contributes to every token at similar wattage, so a big fast "
                 "card and an old slow one look equal.")
        b.append("  wpw(rel) = (compute_share / J-per-token) normalized so the "
                 "LEAST-efficient card = 1.0x — it")
        b.append("  divides each card's energy by the WORK it actually does "
                 "(its --rank-tp-ratio share).")
        pch = (f"{'workload':8s} {'bs':>3s} {'card':22s} {'share':>6s} "
               f"{'W':>6s} {'J/tok':>9s} {'kWh/1M':>9s} {'ct/1M':>9s} "
               f"{'wpw(rel)':>9s}")
        b.append(pch)
        b.append("-" * len(pch))
        for m in result.measurements:
            jl, wl2 = jtok_list(m), watt_list(m)
            rel = _work_per_watt_rel(shares, jl)
            for i, name in enumerate(m.gpu_names or names):
                sh = shares[i] if i < len(shares) and shares[i] is not None else None
                jt = jl[i] if i < len(jl) else 0.0
                w = wl2[i] if i < len(wl2) else 0.0
                rl = rel[i] if i < len(rel) else None
                b.append(
                    f"{m.workload:8s} {m.bucket:3d} {name[:22]:22s} "
                    f"{(f'{sh*100:.0f}%' if sh is not None else '  —'):>6s} "
                    f"{w:6.0f} {jt:9.3f} {jt/3.6:9.3f} {ct1m(jt):9.3f} "
                    f"{(f'{rl:.2f}x' if rl is not None else '  —'):>9s}")
            tot_j = (sum(m.per_card_j_per_prefill_token) if 'PREFILL' in title
                     else sum(m.per_card_j_per_decode_token))
            tot_w = (m.avg_prefill_watts if 'PREFILL' in title else m.avg_decode_watts)
            b.append(
                f"{m.workload:8s} {m.bucket:3d} {'TOTAL (rig)':22s} {'100%':>6s} "
                f"{tot_w:6.0f} {tot_j:9.3f} {tot_j/3.6:9.3f} {ct1m(tot_j):9.3f} "
                f"{'—':>9s}")
        return b

    lines += per_card_block(
        "PER-CARD PREFILL energy (each card's own NVML power, NOT total/N):",
        lambda m: m.per_card_j_per_prefill_token,
        lambda m: m.per_card_avg_prefill_watts)
    lines += per_card_block(
        "PER-CARD DECODE energy (each card's own NVML power, NOT total/N):",
        lambda m: m.per_card_j_per_decode_token,
        lambda m: m.per_card_avg_decode_watts)

    if result.notes:
        lines.append("notes:")
        for n in result.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


def compare_summary(baseline: MeasurementResult, mtp: MeasurementResult,
                    price_ct_per_kwh: float = 30.0) -> str:
    """Side-by-side: no-MTP baseline vs MTP+adaptive, with the measured
    multiplier (decode tok/s ratio) and the energy/cost delta per output token.
    PREFILL and DECODE kept separate."""
    def ct1m(j):
        return (j / 3.6) * price_ct_per_kwh

    def dmap(r):
        return {(m.workload, m.bucket): m for m in r.measurements}
    B, M = dmap(baseline), dmap(mtp)
    lines = ["", "=" * 96,
             "BASELINE (no-MTP) vs MTP+adaptive — decode multiplier + energy/cost per OUTPUT token",
             "=" * 96]
    if mtp.server_avg_accept_length is not None:
        lines.append(f"  server avg accept length: {mtp.server_avg_accept_length:.2f}")
    if mtp.adaptive_step_hist:
        lines.append("  adaptive depth picks (ceiling k=5): " +
                     ", ".join(f"k={k}(x{c})" for k, c in
                               sorted(mtp.adaptive_step_hist.items())))
    hdr = (f"{'workload':8s} {'bs':>3s} | {'base dec':>9s} {'mtp dec':>9s} "
           f"{'x':>5s} {'accept':>7s} | {'base J':>8s} {'mtp J':>8s} "
           f"{'base ct/1M':>10s} {'mtp ct/1M':>10s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for key in sorted(B.keys() & M.keys()):
        b, m = B[key], M[key]
        mult = m.decode_tok_s / b.decode_tok_s if b.decode_tok_s else 0.0
        acc = f"{m.accept_length:.2f}" if m.accept_length is not None else "—"
        lines.append(
            f"{key[0]:8s} {key[1]:3d} | {b.decode_tok_s:9.1f} {m.decode_tok_s:9.1f} "
            f"{mult:4.2f}x {acc:>7s} | {b.j_per_decode_token:8.3f} "
            f"{m.j_per_decode_token:8.3f} {ct1m(b.j_per_decode_token):10.3f} "
            f"{ct1m(m.j_per_decode_token):10.3f}")
    lines.append("(MTP decode is faster AND cheaper per output token: the target "
                 "forward is amortized across accepted tokens.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation run (Qwen3.6-27B-FP8, TP=3 uneven-DCP).
# ---------------------------------------------------------------------------

VALIDATION_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"
DEFAULT_RESULTS_STORE = os.path.join(
    os.path.dirname(__file__), "measured_results.jsonl")


def validation_config(**overrides) -> MeasurementConfig:
    """The no-MTP BASELINE config (Qwen3.6-27B-FP8, TP=3 uneven-DCP)."""
    cfg = MeasurementConfig(
        model_path=VALIDATION_MODEL,
        served_model_name="Qwen3.6-27B",
        tp_size=3,
        rank_gpu_id=[0, 1, 2],                     # cuda:0 = 5090 (torch order)
        rank_tp_ratio=[28591, 16464, 16464],       # 8:4:4 vision-head split
        rank_gpu_memory_mib=[28591, 16464, 16464],
        kv_cache_dtype="fp8_e4m3",
        context_length=8192,
        max_running_requests=32,
        quant_label="fp8",
        label="no-MTP baseline",
        buckets=(1, 4, 16),
        extra_env={"SGLANG_UNEVEN_DCP": "1"},
        port=31000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def mtp_config(**overrides) -> MeasurementConfig:
    """The FASTEST config: NEXTN MTP + adaptive draft (high-accept, ceiling
    k=5, chain topk=1). Same base as the baseline, plus the spec stack."""
    cfg = validation_config(
        label="MTP+adaptive",
        speculative_algorithm="NEXTN",
        speculative_num_steps=5,            # high-accept k=4/5 ladder ceiling
        speculative_eagle_topk=1,           # chain, not tree
        speculative_num_draft_tokens=6,     # num_steps + 1 for the chain
        speculative_adaptive=True,
        speculative_adaptive_config="high-accept",
        # Force OFFLOAD (#93): inactive adaptive states' capture pools are
        # physically unmapped (reserve = MAX over states, not SUM), which is
        # what lets the 5-rung high-accept ladder fit the tight 3080 budget
        # (16464 MiB). 'auto' did NOT engage offload here and OOM'd on the 3080.
        speculative_adaptive_graph_memory="offload",
        # Our top bucket is 16; cap the decode graph ladder there so the
        # per-state capture pools stay small (default followed max_running_reqs
        # up to bs=24, extra pools the 3080 has no room for under 5 states).
        # --enable-memory-saver installs the torch_memory_saver LD_PRELOAD hook
        # that OFFLOAD needs to PHYSICALLY unmap inactive states' capture pools
        # (without it, offload silently keeps all 5 states resident and OOMs the
        # 3080).
        extra_flags=["--cuda-graph-max-bs-decode", "16", "--enable-memory-saver"],
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    p.add_argument("--port", type=int, default=31000)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--buckets", default="1,4,16")
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--store", default=DEFAULT_RESULTS_STORE)
    p.add_argument("--price-ct-per-kwh", type=float, default=30.0)
    p.add_argument("--config", choices=["baseline", "mtp", "both"],
                   default="both", help="which config(s) to measure")
    p.add_argument("--no-ingest", action="store_true")
    args = p.parse_args(argv)

    buckets = tuple(int(x) for x in args.buckets.split(","))
    wls = (
        dataclasses.replace(CODE_WORKLOAD, decode_tokens=args.decode_tokens),
        dataclasses.replace(PROSE_WORKLOAD, decode_tokens=args.decode_tokens),
    )
    common = dict(kv_cache_dtype=args.kv_cache_dtype, port=args.port,
                  context_length=args.context_length, buckets=buckets,
                  workloads=wls)

    todo = []
    if args.config in ("baseline", "both"):
        todo.append(("baseline", validation_config(**common)))
    if args.config in ("mtp", "both"):
        todo.append(("mtp", mtp_config(**common)))

    results = {}
    entries = []
    for name, cfg in todo:
        print(f"\n########## measuring [{cfg.label}] ##########", flush=True)
        r = run_measurement(cfg)
        results[name] = r
        print(summarize(r, args.price_ct_per_kwh))
        entries.extend(r.result_entries())

    if "baseline" in results and "mtp" in results:
        print(compare_summary(results["baseline"], results["mtp"],
                              args.price_ct_per_kwh))

    if not args.no_ingest:
        from sglang.srt.planner.results_store import ResultsStore
        store = ResultsStore.load(args.store)
        for entry in entries:
            store.ingest(entry)
        store.save(args.store)
        print(f"\n[energy] ingested {len(entries)} measured rows -> "
              f"{args.store} (total {len(store)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
