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
"""Sample sources: per-card device state and per-engine statistics.

Two rules run through this module.

**Every field is either a value or a stated reason for its absence.** The
per-rank display DESIGN_216 asks for (SM activity, tensor-pipe activity, DRAM
activity) comes from vendor profiling counters — NVML GPM on NVIDIA, rocprofiler
on AMD — and those are simply not present on much of the hardware this fork
targets. Measured on this rig: ``nvmlGpmQueryDeviceSupport`` reports
``isSupportedDevice = 0`` on both the RTX 3080 and the RTX 5090, because GPM is
a Hopper-and-later feature. A layer that silently substituted the coarse
``utilization.gpu`` percentage for "tensor-pipe activity" would be inventing a
number. So each field carries its :class:`FieldStatus`: available or not,
which source produced it, and — when a coarser source stood in for a finer one
— that the value is a **documented fallback**, which the UI shows rather than
hides. Silent fallback is a recurring defect class in this project; the
telemetry layer is the wrong place to add another one.

**Profiling counters are sampled sparingly.** GPM/DCGM sampling has real
overhead and can collide with an external profiler attached to the same
device. The cadence for those fields is therefore decoupled from the base
cadence (``profile_every`` in :class:`GpuSampler`), and when they are
unavailable the sampler falls back to the cheap NVML utilization rates and
says so.

Energy uses ``nvmlDeviceGetTotalEnergyConsumption`` (a monotonic millijoule
counter) where available, differenced over the window, rather than integrating
power samples — a counter cannot miss a spike between two samples, and J/token
is only as honest as its energy term.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FieldStatus",
    "CardSample",
    "DeviceBackend",
    "NvmlBackend",
    "SmiBackend",
    "NullBackend",
    "GpuSampler",
    "EngineSample",
    "EngineScraper",
    "select_backend",
    "THROTTLE_BITS",
]


# ---------------------------------------------------------------------------
# Field availability
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FieldStatus:
    """Availability of one metric field on this host.

    ``fallback_for`` is set when this field stands in for a finer one that the
    device cannot provide (e.g. NVML utilization instead of GPM SM activity).
    A UI must render that differently from a first-class measurement — it is
    the difference between "34 % of memory bandwidth" and "the memory
    controller was busy 34 % of the sampling window", which are not the same
    statement.
    """

    key: str
    label: str
    unit: str
    available: bool
    source: str
    reason: Optional[str] = None
    fallback_for: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# NVML clock-throttle bitmask (nvmlClocksThrottleReason*). Kept local so the
# module imports without pynvml.
THROTTLE_BITS: Tuple[Tuple[int, str], ...] = (
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "applications_clocks_setting"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal_slowdown"),
    (0x0000000000000040, "hw_thermal_slowdown"),
    (0x0000000000000080, "hw_power_brake_slowdown"),
    (0x0000000000000100, "display_clock_setting"),
)

#: Throttle reasons that actually cost performance. ``gpu_idle`` and the clock
#: *settings* bits are states, not penalties, and must not be reported as
#: "this card is being held back".
PERFORMANCE_THROTTLES = frozenset(
    {
        "sw_power_cap",
        "hw_slowdown",
        "sw_thermal_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
)


def decode_throttle(mask: Optional[int]) -> List[str]:
    if mask is None:
        return []
    return [name for bit, name in THROTTLE_BITS if mask & bit]


# ---------------------------------------------------------------------------
# Card sample
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CardSample:
    """One card at one instant. ``None`` means "not available on this host";
    the accompanying :meth:`GpuSampler.field_report` says why."""

    index: int
    name: str
    uuid: Optional[str] = None

    mem_total_mib: Optional[int] = None
    mem_used_mib: Optional[int] = None

    temp_c: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    #: Monotonic total-energy counter (millijoule) — differenced for J/token.
    energy_mj: Optional[int] = None

    sm_clock_mhz: Optional[int] = None
    sm_clock_max_mhz: Optional[int] = None
    mem_clock_mhz: Optional[int] = None
    pstate: Optional[int] = None
    throttle: List[str] = dataclasses.field(default_factory=list)

    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None

    #: Coarse NVML utilization: fraction of the window in which at least one
    #: kernel was resident (``gpu``) / the memory controller was busy
    #: (``memory``). NOT occupancy and NOT achieved bandwidth.
    util_gpu_pct: Optional[float] = None
    util_mem_pct: Optional[float] = None

    #: Fine profiling counters (NVML GPM / DCGM). Absent on consumer cards.
    sm_active: Optional[float] = None
    tensor_active: Optional[float] = None
    dram_active: Optional[float] = None
    #: Which source produced the activity fields for THIS sample.
    activity_source: str = "none"

    def clock_ratio(self) -> Optional[float]:
        if not self.sm_clock_mhz or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz

    def performance_throttles(self) -> List[str]:
        return [t for t in self.throttle if t in PERFORMANCE_THROTTLES]

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Device backends
# ---------------------------------------------------------------------------


class DeviceBackend:
    """Vendor-neutral device interface.

    An implementation reports what it can read (:meth:`fields`) and produces
    samples (:meth:`sample`). A ROCm implementation over ``rocm-smi`` /
    ``rocprofiler`` slots in here without any change above this line; until one
    exists, an AMD host resolves to :class:`NullBackend`, which is honest
    (every field unavailable, with a reason) rather than empty.
    """

    name = "abstract"

    def fields(self) -> List[FieldStatus]:
        raise NotImplementedError

    def sample(self, with_profiling: bool = False) -> List[CardSample]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullBackend(DeviceBackend):
    """No device telemetry available. Carries the reason instead of pretending
    the rig has no cards."""

    name = "none"

    def __init__(self, reason: str):
        self.reason = reason

    def fields(self) -> List[FieldStatus]:
        return [
            FieldStatus(
                key="*",
                label="all device fields",
                unit="",
                available=False,
                source="none",
                reason=self.reason,
            )
        ]

    def sample(self, with_profiling: bool = False) -> List[CardSample]:
        return []


class NvmlBackend(DeviceBackend):
    """pynvml. Read-only: no CUDA context is created, so this is safe to run
    beside a live server and beside another agent's exclusive GPU work."""

    name = "pynvml"

    def __init__(self, pynvml_module=None):
        self._n = pynvml_module
        if self._n is None:
            import pynvml as _n  # noqa: PLC0415  (optional dependency)

            self._n = _n
        self._n.nvmlInit()
        self._handles: List[Tuple[int, Any]] = []
        for i in range(self._n.nvmlDeviceGetCount()):
            self._handles.append((i, self._n.nvmlDeviceGetHandleByIndex(i)))
        self._gpm_reason: Optional[str] = None
        self._gpm_supported = self._probe_gpm()
        self._energy_reason: Optional[str] = None
        self._energy_supported = self._probe_energy()

    # -- capability probes (run once, at construction) ----------------------

    def _probe_gpm(self) -> bool:
        n = self._n
        if not hasattr(n, "nvmlGpmQueryDeviceSupport"):
            self._gpm_reason = (
                "the installed pynvml has no GPM bindings "
                "(nvmlGpmQueryDeviceSupport missing)"
            )
            return False
        for _, h in self._handles:
            try:
                if not n.nvmlGpmQueryDeviceSupport(h).isSupportedDevice:
                    self._gpm_reason = (
                        "NVML GPM reports isSupportedDevice=0 for at least one "
                        "card: GPM profiling counters are a Hopper-and-later "
                        "feature and are not exposed on consumer GeForce parts"
                    )
                    return False
            except Exception as e:
                self._gpm_reason = f"NVML GPM query failed ({type(e).__name__}: {e})"
                return False
        return bool(self._handles)

    def _probe_energy(self) -> bool:
        n = self._n
        for _, h in self._handles:
            try:
                n.nvmlDeviceGetTotalEnergyConsumption(h)
            except Exception as e:
                self._energy_reason = (
                    f"nvmlDeviceGetTotalEnergyConsumption unsupported "
                    f"({type(e).__name__}); energy falls back to integrating "
                    "power samples, which can miss transients between samples"
                )
                return False
        return bool(self._handles)

    # -- reporting ----------------------------------------------------------

    def fields(self) -> List[FieldStatus]:
        f = [
            FieldStatus("mem_used_mib", "VRAM used", "MiB", True, "nvml"),
            FieldStatus("temp_c", "temperature", "C", True, "nvml"),
            FieldStatus("power_w", "power draw", "W", True, "nvml"),
            FieldStatus("sm_clock_mhz", "SM clock", "MHz", True, "nvml"),
            FieldStatus("pstate", "performance state", "", True, "nvml"),
            FieldStatus("throttle", "throttle reasons", "", True, "nvml"),
            FieldStatus(
                "util_gpu_pct",
                "GPU busy share",
                "%",
                True,
                "nvml-utilization",
                reason=(
                    "coarse: share of the sampling window with at least one "
                    "resident kernel; not SM occupancy"
                ),
            ),
            FieldStatus(
                "util_mem_pct",
                "memory-controller busy share",
                "%",
                True,
                "nvml-utilization",
                reason=(
                    "coarse: share of the window with memory traffic; not "
                    "achieved bandwidth"
                ),
            ),
        ]
        f.append(
            FieldStatus(
                "energy_mj",
                "energy counter",
                "mJ",
                self._energy_supported,
                "nvml" if self._energy_supported else "power-integration",
                reason=None if self._energy_supported else self._energy_reason,
                fallback_for=None if self._energy_supported else "energy_mj",
            )
        )
        for key, label in (
            ("sm_active", "SM activity"),
            ("tensor_active", "tensor-pipe activity"),
            ("dram_active", "DRAM activity"),
        ):
            if self._gpm_supported:
                f.append(FieldStatus(key, label, "fraction", True, "nvml-gpm"))
            else:
                f.append(
                    FieldStatus(
                        key,
                        label,
                        "fraction",
                        False,
                        "nvml-gpm",
                        reason=self._gpm_reason,
                    )
                )
        if not self._gpm_supported:
            f.append(
                FieldStatus(
                    "activity_fallback",
                    "activity via coarse utilization",
                    "%",
                    True,
                    "nvml-utilization",
                    reason=(
                        "GPM unavailable, so SM/DRAM activity is approximated "
                        "by nvmlDeviceGetUtilizationRates; the tensor pipe has "
                        "no coarse equivalent and stays unavailable"
                    ),
                    fallback_for="sm_active,dram_active",
                )
            )
        return f

    # -- sampling -----------------------------------------------------------

    @staticmethod
    def _try(fn, *a, default=None):
        try:
            return fn(*a)
        except Exception:
            return default

    def sample(self, with_profiling: bool = False) -> List[CardSample]:
        n = self._n
        out: List[CardSample] = []
        for i, h in self._handles:
            mem = self._try(n.nvmlDeviceGetMemoryInfo, h)
            util = self._try(n.nvmlDeviceGetUtilizationRates, h)
            name = self._try(n.nvmlDeviceGetName, h, default="unknown")
            if isinstance(name, bytes):
                name = name.decode()
            uuid = self._try(n.nvmlDeviceGetUUID, h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            mask = self._try(
                getattr(n, "nvmlDeviceGetCurrentClocksThrottleReasons", None) or (lambda *_: None),
                h,
            )
            if mask is None:
                mask = self._try(
                    getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None)
                    or (lambda *_: None),
                    h,
                )
            power = self._try(n.nvmlDeviceGetPowerUsage, h)
            limit = self._try(n.nvmlDeviceGetEnforcedPowerLimit, h)
            cs = CardSample(
                index=i,
                name=str(name),
                uuid=uuid,
                mem_total_mib=(mem.total // 2**20) if mem else None,
                mem_used_mib=(mem.used // 2**20) if mem else None,
                temp_c=self._try(
                    n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU
                ),
                power_w=(power / 1000.0) if power is not None else None,
                power_limit_w=(limit / 1000.0) if limit is not None else None,
                energy_mj=(
                    self._try(n.nvmlDeviceGetTotalEnergyConsumption, h)
                    if self._energy_supported
                    else None
                ),
                sm_clock_mhz=self._try(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM),
                sm_clock_max_mhz=self._try(
                    n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_SM
                ),
                mem_clock_mhz=self._try(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM),
                pstate=self._try(n.nvmlDeviceGetPerformanceState, h),
                throttle=decode_throttle(mask),
                pcie_gen=self._try(n.nvmlDeviceGetCurrPcieLinkGeneration, h),
                pcie_width=self._try(n.nvmlDeviceGetCurrPcieLinkWidth, h),
                util_gpu_pct=(float(util.gpu) if util is not None else None),
                util_mem_pct=(float(util.memory) if util is not None else None),
            )
            if self._gpm_supported and with_profiling:
                self._fill_gpm(h, cs)
            elif util is not None:
                # Documented fallback: coarse utilization stands in for SM and
                # DRAM activity. The tensor pipe has NO coarse equivalent and
                # deliberately stays None rather than being faked from util.
                cs.sm_active = float(util.gpu) / 100.0
                cs.dram_active = float(util.memory) / 100.0
                cs.activity_source = "nvml-utilization (coarse fallback)"
            out.append(cs)
        return out

    def _fill_gpm(self, handle, cs: CardSample) -> None:
        """GPM metric read. Sampled only when asked (``with_profiling``), since
        the counters carry overhead and can collide with an attached
        profiler."""
        n = self._n
        try:
            s0 = n.nvmlGpmSampleAlloc()
            s1 = n.nvmlGpmSampleAlloc()
            n.nvmlGpmSampleGet(handle, s0)
            time.sleep(0.05)
            n.nvmlGpmSampleGet(handle, s1)
            g = n.c_nvmlGpmMetricsGet_t()
            g.version = n.NVML_GPM_METRICS_GET_VERSION
            g.numMetrics = 3
            g.sample1 = s0
            g.sample2 = s1
            g.metrics[0].metricId = n.NVML_GPM_METRIC_SM_OCCUPANCY
            g.metrics[1].metricId = n.NVML_GPM_METRIC_ANY_TENSOR_UTIL
            g.metrics[2].metricId = n.NVML_GPM_METRIC_DRAM_BW_UTIL
            n.nvmlGpmMetricsGet(g)
            cs.sm_active = g.metrics[0].value / 100.0
            cs.tensor_active = g.metrics[1].value / 100.0
            cs.dram_active = g.metrics[2].value / 100.0
            cs.activity_source = "nvml-gpm"
            n.nvmlGpmSampleFree(s0)
            n.nvmlGpmSampleFree(s1)
        except Exception as e:
            self._gpm_supported = False
            self._gpm_reason = f"GPM read failed at runtime ({type(e).__name__}: {e})"
            cs.activity_source = "none"

    def close(self) -> None:
        try:
            self._n.nvmlShutdown()
        except Exception:
            pass


class SmiBackend(DeviceBackend):
    """``nvidia-smi`` CSV fallback for hosts without pynvml.

    Deliberately reduced: no energy counter, no GPM. Everything missing is
    reported as missing.
    """

    name = "nvidia-smi"

    QUERY = (
        "index,name,uuid,memory.total,memory.used,temperature.gpu,"
        "utilization.gpu,utilization.memory,power.draw,enforced.power.limit,"
        "clocks.sm,clocks.max.sm,clocks.mem,pstate,"
        "clocks_throttle_reasons.active,"
        "pcie.link.gen.current,pcie.link.width.current"
    )

    def __init__(self, runner: Optional[Callable[[List[str]], str]] = None):
        self._run = runner or (
            lambda cmd: subprocess.check_output(cmd, text=True, timeout=10)
        )
        # Fail loudly at construction if the binary is not there, so
        # select_backend can move on to NullBackend with a reason.
        self._run(["nvidia-smi", "--version"])

    def fields(self) -> List[FieldStatus]:
        f = [
            FieldStatus("mem_used_mib", "VRAM used", "MiB", True, "nvidia-smi"),
            FieldStatus("temp_c", "temperature", "C", True, "nvidia-smi"),
            FieldStatus("power_w", "power draw", "W", True, "nvidia-smi"),
            FieldStatus("sm_clock_mhz", "SM clock", "MHz", True, "nvidia-smi"),
            FieldStatus("throttle", "throttle reasons", "", True, "nvidia-smi"),
            FieldStatus(
                "util_gpu_pct", "GPU busy share", "%", True, "nvidia-smi",
                reason="coarse: not SM occupancy",
            ),
            FieldStatus(
                "energy_mj", "energy counter", "mJ", False, "nvidia-smi",
                reason=(
                    "nvidia-smi exposes no total-energy counter; install "
                    "pynvml for nvmlDeviceGetTotalEnergyConsumption"
                ),
            ),
        ]
        for key, label in (
            ("sm_active", "SM activity"),
            ("tensor_active", "tensor-pipe activity"),
            ("dram_active", "DRAM activity"),
        ):
            f.append(
                FieldStatus(
                    key, label, "fraction", False, "nvml-gpm",
                    reason="profiling counters need pynvml GPM or DCGM",
                )
            )
        return f

    @staticmethod
    def _num(v, cast=float):
        try:
            return cast(float(v))
        except (TypeError, ValueError):
            return None

    def sample(self, with_profiling: bool = False) -> List[CardSample]:
        txt = self._run(
            [
                "nvidia-smi",
                f"--query-gpu={self.QUERY}",
                "--format=csv,noheader,nounits",
            ]
        )
        out = []
        for line in txt.strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            if len(c) < 17:
                continue
            util_gpu = self._num(c[6])
            util_mem = self._num(c[7])
            try:
                mask = int(c[14], 16) if c[14].lower().startswith("0x") else None
            except ValueError:
                mask = None
            pstate = c[13]
            out.append(
                CardSample(
                    index=self._num(c[0], int) or 0,
                    name=c[1],
                    uuid=c[2],
                    mem_total_mib=self._num(c[3], int),
                    mem_used_mib=self._num(c[4], int),
                    temp_c=self._num(c[5]),
                    util_gpu_pct=util_gpu,
                    util_mem_pct=util_mem,
                    power_w=self._num(c[8]),
                    power_limit_w=self._num(c[9]),
                    sm_clock_mhz=self._num(c[10], int),
                    sm_clock_max_mhz=self._num(c[11], int),
                    mem_clock_mhz=self._num(c[12], int),
                    pstate=(
                        self._num(pstate[1:], int) if pstate.startswith("P") else None
                    ),
                    throttle=decode_throttle(mask),
                    pcie_gen=self._num(c[15], int),
                    pcie_width=self._num(c[16], int),
                    sm_active=(util_gpu / 100.0 if util_gpu is not None else None),
                    dram_active=(util_mem / 100.0 if util_mem is not None else None),
                    activity_source="nvidia-smi utilization (coarse fallback)",
                )
            )
        return out


def select_backend(prefer: str = "auto", pynvml_module=None) -> DeviceBackend:
    """pynvml -> nvidia-smi -> NullBackend(reason). Never raises: a host
    without GPUs is a valid aggregator host."""
    reasons = []
    if prefer in ("auto", "pynvml"):
        try:
            return NvmlBackend(pynvml_module)
        except Exception as e:
            reasons.append(f"pynvml: {type(e).__name__}: {e}")
    if prefer in ("auto", "nvidia-smi"):
        try:
            return SmiBackend()
        except Exception as e:
            reasons.append(f"nvidia-smi: {type(e).__name__}: {e}")
    return NullBackend("; ".join(reasons) or f"backend {prefer!r} not built")


# ---------------------------------------------------------------------------
# GPU sampler
# ---------------------------------------------------------------------------


class GpuSampler:
    """Wraps a backend with the profiling-cadence policy and the field report.

    ``profile_every`` decouples the (expensive, collision-prone) profiling
    counters from the base cadence: with a 1 s base and ``profile_every=10``
    the GPM read happens every ten seconds. Set to 0 to never read them.
    """

    def __init__(self, backend: Optional[DeviceBackend] = None, profile_every: int = 10):
        self.backend = backend if backend is not None else select_backend()
        self.profile_every = max(0, int(profile_every))
        self._tick = 0
        self._fields: Optional[List[FieldStatus]] = None

    def field_report(self) -> List[FieldStatus]:
        if self._fields is None:
            try:
                self._fields = self.backend.fields()
            except Exception as e:
                self._fields = [
                    FieldStatus(
                        "*", "all device fields", "", False, self.backend.name,
                        reason=f"{type(e).__name__}: {e}",
                    )
                ]
        return self._fields

    def sample(self) -> List[CardSample]:
        want_profiling = bool(
            self.profile_every and self._tick % self.profile_every == 0
        )
        self._tick += 1
        try:
            return self.backend.sample(with_profiling=want_profiling)
        except Exception:
            # A transient NVML failure must not kill the collector loop; the
            # gap shows up as a bucket with n=0 rather than as a crash.
            return []


# ---------------------------------------------------------------------------
# Engine statistics
# ---------------------------------------------------------------------------

_METRIC = re.compile(r"^(sglang:[a-z_0-9]+)(?:\{([^}]*)\})?\s+([0-9eE.+-]+)\s*$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

#: Metrics the engine already publishes PER TP RANK, which is what makes an
#: exact per-rank work attribution possible without any profiling counter.
#: ``forward_execution_seconds_total`` is GPU-busy time, so differencing it
#: gives a rank's active share directly; the estimated FLOP and byte counters
#: are derived from the shapes actually executed, so differencing them gives
#: achieved rates that can be put against the probe's measured peaks.
#:
#: Caveat that must travel with them: unless the server runs with
#: ``--enable-metrics-for-all-schedulers`` only TP rank 0 exports, and every
#: series then carries ``tp_rank="0"`` regardless of which rank did the work.
PER_RANK_KEYS = {
    "forward_execution_seconds_total": "sglang:forward_execution_seconds_total",
    "estimated_flops_per_gpu_total": "sglang:estimated_flops_per_gpu_total",
    "estimated_read_bytes_per_gpu_total": "sglang:estimated_read_bytes_per_gpu_total",
    "estimated_write_bytes_per_gpu_total": "sglang:estimated_write_bytes_per_gpu_total",
    "fwd_occupancy": "sglang:fwd_occupancy",
}

#: Prometheus names the collector keeps, mapped to short keys.
_ENGINE_KEYS = {
    "gen_throughput": "sglang:gen_throughput",
    "num_running_reqs": "sglang:num_running_reqs",
    "num_queue_reqs": "sglang:num_queue_reqs",
    "num_paused_reqs": "sglang:num_paused_reqs",
    "token_usage": "sglang:token_usage",
    "kv_used_tokens": "sglang:kv_used_tokens",
    "kv_available_tokens": "sglang:kv_available_tokens",
    "mamba_used_tokens": "sglang:mamba_used_tokens",
    "max_total_num_tokens": "sglang:max_total_num_tokens",
    "cache_hit_rate": "sglang:cache_hit_rate",
    "spec_accept_length": "sglang:spec_accept_length",
    "num_requests_total": "sglang:num_requests_total",
    "prompt_tokens_total": "sglang:prompt_tokens_total",
    "generation_tokens_total": "sglang:generation_tokens_total",
    "e2e_request_latency_seconds": "sglang:e2e_request_latency_seconds_sum",
    "inter_token_latency_seconds": "sglang:inter_token_latency_seconds_sum",
    "utilization": "sglang:utilization",
}


def parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    """Parse an exposition into ``name -> [(labels, value), ...]``.

    Labels are KEPT. An earlier version of this parser collapsed each metric to
    its bare name and retained the last value seen, which silently discarded
    exactly the dimension that matters here: the engine labels its counters
    with ``tp_rank``, so collapsing them threw away the per-rank breakdown and
    replaced it with whichever rank happened to be printed last.
    """
    out: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        labels = dict(_LABEL.findall(m.group(2) or ""))
        out.setdefault(m.group(1), []).append((labels, value))
    return out


def _sum_series(series: Optional[List[Tuple[Dict[str, str], float]]]) -> Optional[float]:
    """Total a counter across its label sets.

    Summing is right for counters split by a secondary label (a forward-time
    counter carries a ``category`` label per forward mode, and the rank's busy
    time is the total across modes). Picking one arbitrarily would under-report.
    """
    if not series:
        return None
    return sum(v for _, v in series)


@dataclasses.dataclass
class EngineSample:
    """The engine side of the join. ``up`` false with a ``reason`` is a valid
    sample — a dashboard that shows a gap must be able to say whether the
    server was down or the collector was."""

    up: bool
    reason: Optional[str] = None
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    info: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: tp_rank -> {short key: cumulative value}. Empty when the server does not
    #: export per-rank series.
    per_rank: Dict[int, Dict[str, float]] = dataclasses.field(default_factory=dict)
    #: Set when only one rank exports, i.e. the per-rank view is really the
    #: rank-0 view wearing every rank's name.
    per_rank_note: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


class EngineScraper:
    """Reads ``/metrics`` and ``/get_server_info`` from a local engine.

    This lives on the host, next to the engine, for the reason DESIGN_216
    gives: tok/s is not an NVML quantity, and joining it with card state
    requires a process that can see both.
    """

    def __init__(self, base_url: str = "", timeout: float = 1.5, opener=None):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self._open = opener or self._urlopen

    def _urlopen(self, url: str) -> str:
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return r.read().decode("utf-8", "replace")

    def scrape(self) -> EngineSample:
        if not self.base_url:
            return EngineSample(up=False, reason="no engine URL configured")
        try:
            txt = self._open(self.base_url + "/metrics")
        except Exception as e:
            return EngineSample(
                up=False, reason=f"{self.base_url}/metrics unreachable ({e})"
            )
        parsed = parse_prometheus(txt)
        metrics: Dict[str, float] = {}
        for short, name in _ENGINE_KEYS.items():
            total = _sum_series(parsed.get(name))
            if total is not None:
                metrics[short] = total

        per_rank: Dict[int, Dict[str, float]] = {}
        for short, name in PER_RANK_KEYS.items():
            for labels, value in parsed.get(name, []):
                raw = labels.get("tp_rank")
                if raw is None:
                    continue
                try:
                    rank = int(raw)
                except ValueError:
                    continue
                slot = per_rank.setdefault(rank, {})
                # Counters split by a secondary label (category / mode)
                # accumulate; gauges simply take the latest value.
                if short.endswith("_total"):
                    slot[short] = slot.get(short, 0.0) + value
                else:
                    slot[short] = value

        note = None
        if len(per_rank) == 1:
            note = (
                "only one TP rank exports metrics, so this is rank "
                f"{next(iter(per_rank))}'s view, not a per-rank breakdown. "
                "Start the server with --enable-metrics-for-all-schedulers to "
                "get one series per rank."
            )
        elif not per_rank:
            note = (
                "the server exports no per-rank series; per-rank work "
                "attribution falls back to device counters"
            )

        info: Dict[str, Any] = {}
        for path in ("/get_server_info", "/server_info"):
            try:
                info = json.loads(self._open(self.base_url + path))
                break
            except Exception:
                continue
        return EngineSample(
            up=True,
            metrics=metrics,
            info=info,
            per_rank=per_rank,
            per_rank_note=note,
        )
