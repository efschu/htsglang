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
            c += ["--rank-tp-ratio", ",".join(map(str, self.rank_tp_ratio))]
        if self.rank_gpu_memory_mib is not None:
            c += ["--rank-gpu-memory-mib",
                  ",".join(map(str, self.rank_gpu_memory_mib))]
        if self.trust_remote_code:
            c += ["--trust-remote-code"]
        if self.disable_custom_all_reduce:
            c += ["--disable-custom-all-reduce"]
        if self.disable_cuda_graph:
            c += ["--disable-cuda-graph"]
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
        t_done = time.time()
        if t_first is None:
            t_first = t_done
        return {
            "t_send": t_send, "t_first": t_first, "t_done": t_done,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
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
                print(
                    f"[energy] {wl.name:5s} bs={bucket:<3d} "
                    f"prefill {bm.prefill_tok_s:8.1f} tok/s "
                    f"{bm.j_per_prefill_token:6.3f} J/tok | "
                    f"decode {bm.decode_tok_s:7.1f} tok/s "
                    f"{bm.j_per_decode_token:6.3f} J/tok "
                    f"({bm.avg_decode_watts:.0f} W)", flush=True)

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
            compute_share_by_card=share,
            notes=self.notes,
        )

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


def summarize(result: MeasurementResult) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("MEASURED energy + throughput (CUDA graphs ON, temp 0)")
    lines.append(f"  rig: {', '.join(f'{c}x {n}' for c, n, m in result.hardware_cards)}")
    lines.append(f"  idle whole-rig: {result.idle_watts:.0f} W")
    lines.append("  energy = GPU (NVML) board power only, summed across cards — "
                 "EXCLUDES CPU/RAM/PSU losses (NOT wall-socket power).")
    if result.compute_share_by_card:
        shares = ", ".join(
            f"{n} {s*100:.0f}%" for n, s in
            zip(result.gpu_names_sampled, result.compute_share_by_card))
        lines.append(f"  uneven-TP compute share (from ratio): {shares}")
    lines.append("=" * 78)
    header = (f"{'workload':8s} {'bs':>3s} {'prefill tok/s':>13s} "
              f"{'J/pretok':>9s} {'kWh/1M':>8s} | {'decode tok/s':>12s} "
              f"{'J/dectok':>9s} {'kWh/1M':>8s} {'dec W':>7s}")
    lines.append(header)
    lines.append("-" * len(header))
    for m in result.measurements:
        lines.append(
            f"{m.workload:8s} {m.bucket:3d} {m.prefill_tok_s:13.1f} "
            f"{m.j_per_prefill_token:9.3f} {m.kwh_per_1m_prefill():8.2f} | "
            f"{m.decode_tok_s:12.1f} {m.j_per_decode_token:9.3f} "
            f"{m.kwh_per_1m_decode():8.2f} {m.avg_decode_watts:7.0f}")
    # -- PER-CARD decode breakdown (each card's OWN measured power) ----------
    lines.append("")
    lines.append("PER-CARD decode energy (each card's own NVML power, NOT total/N):")
    names = result.gpu_names_sampled
    shares = result.compute_share_by_card or [None] * len(names)
    pch = (f"{'workload':8s} {'bs':>3s} {'card':22s} {'share':>6s} "
           f"{'dec W':>7s} {'J/dectok':>9s} {'kWh/1M':>8s}")
    lines.append(pch)
    lines.append("-" * len(pch))
    for m in result.measurements:
        for i, name in enumerate(m.gpu_names or names):
            sh = shares[i] if i < len(shares) and shares[i] is not None else None
            jd = m.per_card_j_per_decode_token[i] if i < len(m.per_card_j_per_decode_token) else 0.0
            w = m.per_card_avg_decode_watts[i] if i < len(m.per_card_avg_decode_watts) else 0.0
            lines.append(
                f"{m.workload:8s} {m.bucket:3d} {name[:22]:22s} "
                f"{(f'{sh*100:.0f}%' if sh is not None else '  —'):>6s} "
                f"{w:7.0f} {jd:9.3f} {jd/3.6:8.2f}")
        lines.append(
            f"{m.workload:8s} {m.bucket:3d} {'TOTAL (rig)':22s} {'100%':>6s} "
            f"{m.avg_decode_watts:7.0f} {m.j_per_decode_token:9.3f} "
            f"{m.kwh_per_1m_decode():8.2f}")
    if result.notes:
        lines.append("notes:")
        for n in result.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation run (Qwen3.6-27B-FP8, TP=3 uneven-DCP).
# ---------------------------------------------------------------------------

VALIDATION_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"
DEFAULT_RESULTS_STORE = os.path.join(
    os.path.dirname(__file__), "measured_results.jsonl")


def validation_config(**overrides) -> MeasurementConfig:
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
        buckets=(1, 4, 16),
        extra_env={"SGLANG_UNEVEN_DCP": "1"},
        port=31000,
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
    p.add_argument("--no-ingest", action="store_true")
    args = p.parse_args(argv)

    buckets = tuple(int(x) for x in args.buckets.split(","))
    wls = (
        dataclasses.replace(CODE_WORKLOAD, decode_tokens=args.decode_tokens),
        dataclasses.replace(PROSE_WORKLOAD, decode_tokens=args.decode_tokens),
    )
    cfg = validation_config(
        kv_cache_dtype=args.kv_cache_dtype, port=args.port,
        context_length=args.context_length, buckets=buckets, workloads=wls)

    result = run_measurement(cfg)
    print(summarize(result))

    if not args.no_ingest:
        from sglang.srt.planner.results_store import ResultsStore
        store = ResultsStore.load(args.store)
        for entry in result.result_entries():
            store.ingest(entry)
        store.save(args.store)
        print(f"\n[energy] ingested {len(result.result_entries())} measured "
              f"rows -> {args.store} (total {len(store)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
