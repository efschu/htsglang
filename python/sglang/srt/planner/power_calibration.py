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
"""Per-card POWER CALIBRATION — a reusable, frontend-triggerable backend probe.

WHAT THIS IS (and is NOT)
-------------------------
This is a REUSABLE measurement backend the dashboard calls (later, from a
button — the button wiring lives in webui.py, not here). It does NOT hardcode
this box's watt numbers as constants. It MEASURES real per-card board power on
whatever rig it runs on and persists the result as a MEASURED artifact
(``provenance="measured"``). The energy model (``roofline.py``) then consumes
that table as an OVERRIDE for the real cards it recognises (by UUID/arch),
falling back to the ``IDLE_FRACTION_OF_TDP`` heuristic ONLY for virtual /
unmeasured cards. This mirrors the fork's existing "measured overrides
estimate" discipline (uneven_perf's GEMM/membw probe already wins over the
nameplate peaks; this is the power sibling of that).

WHAT IT MEASURES (per physical card, one at a time)
---------------------------------------------------
Board power (``nvmlDeviceGetPowerUsage``, sampled ~50-100 ms and averaged over
the window) in three states:

  1. ``p_idle_w``  — CUDA context alive, a small tensor resident, NO active
     kernel. This is the lockstep-WAIT floor a TP rank sits at while blocked on
     a collective, NOT desktop idle (which is a few W lower).
  2. ``p_membw_w`` — during a sustained pure-bandwidth kernel (the decode-
     relevant, memory-bound active power).
  3. ``p_gemm_w``  — during a sustained large GEMM at high occupancy (the
     prefill-relevant, compute-bound active power).

It also records the throughput the benches return (``membw_gbs`` /
``gemm_tflops``) so the same artifact doubles as a perf provenance if wanted.

The two active kernels are REUSED verbatim from ``uneven_perf`` (``_bench_
membw_gbs`` / ``_bench_gemm_tflops``) — the same kernels the roofline's membw /
GEMM peaks come from, so the power is attributed to exactly the workload the
throughput number describes. Kernel inputs are sampled with ``torch.randn`` on
the device inside those benches (bf16); no host->device input staging is needed
because they are self-contained.

DEVICE-ORDER TRAP + ISOLATION (shared single box)
-------------------------------------------------
torch ``cuda:0`` is NOT NVML index 0 (torch orders by its own enumeration,
NVML by PCI bus). Every watt here is attributed to the PHYSICAL card by NVML
UUID, never by a bare index. Each card is measured in its OWN subprocess with
``CUDA_VISIBLE_DEVICES=<uuid>`` so exactly one physical GPU is visible to torch
(``cuda:0`` is then unambiguous); the ``PowerSampler`` inside that subprocess
still resolves the card's real NVML handle by UUID (NVML enumerates every card
regardless of ``CUDA_VISIBLE_DEVICES``) and samples only that one board.

Before touching a card we check ``nvidia-smi --query-compute-apps``: if another
process is already using it, we SKIP it and report — we never fight for it and
never broad-``pkill`` (other agents share this box). We only ever spawn and reap
our own subprocesses. Each micro-benchmark window is short (~4-5 s) and frees
its CUDA context on exit.

IMPORT-SAFE WITHOUT A GPU
-------------------------
Module import pulls in no torch / pynvml (both are imported lazily inside the
measurement functions), so the parser / persistence / roofline-consumption code
is unit-testable on a box with no GPU. Only ``measure_card_power`` /
``measure_all_cards`` touch hardware.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

__all__ = [
    "CardPowerMeasurement",
    "PowerCalibrationResult",
    "DEFAULT_POWER_PROFILE_PATH",
    "measure_card_power",
    "measure_all_cards",
    "save_power_profile",
    "load_power_profile",
    "power_profile_by_arch",
    "busy_card_uuids",
]

#: Canonical persistence path — sits next to uneven_perf's ``hw_profile-*.json``
#: probe cache, session-persistent, keyed by card UUID. A fresh run overwrites
#: it (re-runnable). Overridable per call for tests.
DEFAULT_POWER_PROFILE_PATH = os.path.expanduser("~/.cache/sglang/power_profile.json")

#: Measured-artifact provenance tag (same token results_store treats as a real
#: measurement, and the opposite of roofline's "planner-estimate"). Consuming
#: code uses this to know the numbers are read from NVML, not modelled.
MEASURED_PROVENANCE = "measured"

#: Default sampling cadence + window lengths (seconds). ~50 ms cadence matches
#: the task's 50-100 ms guidance; a short warmup lets clocks ramp before the
#: measurement window, which is kept short (shared-box discipline).
_DEFAULT_SAMPLE_MS = 50.0
_DEFAULT_IDLE_WINDOW_S = 2.0
_DEFAULT_ACTIVE_WARMUP_S = 1.0
_DEFAULT_ACTIVE_WINDOW_S = 4.0


# ---------------------------------------------------------------------------
# Result types (pure data — no GPU needed to build / parse these).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardPowerMeasurement:
    """MEASURED per-physical-card power table row, keyed by ``uuid`` + ``arch``.

    All three watt anchors are averaged NVML board power over the measurement
    window for that state. This is GPU-only board power (excludes CPU/RAM/PSU
    conversion — NOT wall-socket power)."""

    uuid: str
    name: str
    arch: str  # sm-arch, e.g. "sm120" (5090) / "sm86" (3080)
    total_mib: int
    p_idle_w: float
    p_membw_w: float
    p_gemm_w: float
    membw_gbs: float
    gemm_tflops: float
    provenance: str = MEASURED_PROVENANCE
    driver: Optional[str] = None
    measured_at: Optional[str] = None
    sample_ms: float = _DEFAULT_SAMPLE_MS
    window_s: float = _DEFAULT_ACTIVE_WINDOW_S

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CardPowerMeasurement":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclasses.dataclass
class PowerCalibrationResult:
    """The whole-rig calibration run: one measurement per card actually probed,
    plus the cards that were SKIPPED (busy with another process, or errored)."""

    cards: List[CardPowerMeasurement]
    skipped: List[dict] = dataclasses.field(default_factory=list)
    driver: Optional[str] = None
    created: Optional[str] = None

    def by_uuid(self) -> Dict[str, CardPowerMeasurement]:
        return {c.uuid: c for c in self.cards}

    def to_json(self) -> dict:
        return {
            "provenance": MEASURED_PROVENANCE,
            "driver": self.driver,
            "created": self.created,
            "cards": [c.to_json() for c in self.cards],
            "skipped": list(self.skipped),
        }

    @classmethod
    def from_json(cls, d: dict) -> "PowerCalibrationResult":
        return cls(
            cards=[CardPowerMeasurement.from_json(c) for c in d.get("cards", [])],
            skipped=list(d.get("skipped", [])),
            driver=d.get("driver"),
            created=d.get("created"),
        )


# ---------------------------------------------------------------------------
# Persistence + consumption helpers (pure — safe without a GPU).
# ---------------------------------------------------------------------------


def save_power_profile(
    result: PowerCalibrationResult, path: str = DEFAULT_POWER_PROFILE_PATH
) -> str:
    """Persist the MEASURED table atomically. Returns the path written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result.to_json(), f, indent=1)
    os.replace(tmp, path)
    return path


def load_power_profile(
    path: str = DEFAULT_POWER_PROFILE_PATH,
) -> Dict[str, CardPowerMeasurement]:
    """Load the persisted MEASURED power table as ``{uuid: CardPowerMeasurement}``.

    Returns ``{}`` when the file is absent or unreadable — the consumer then
    simply falls back to the heuristic. NEVER raises (best-effort read)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    try:
        res = PowerCalibrationResult.from_json(data)
    except Exception:
        return {}
    return {c.uuid: c for c in res.cards if c.provenance == MEASURED_PROVENANCE}


def power_profile_by_arch(
    profile: Dict[str, CardPowerMeasurement]
) -> Dict[str, CardPowerMeasurement]:
    """Index a ``{uuid: measurement}`` map by sm-arch as a fallback key (used
    when a card's UUID is not directly in the table but its arch is — identical
    silicon draws the same power). If several cards share an arch, the first
    wins; they are structurally the same card so their anchors coincide."""
    out: Dict[str, CardPowerMeasurement] = {}
    for m in profile.values():
        out.setdefault(m.arch, m)
    return out


# ---------------------------------------------------------------------------
# Busy-card guard (shared box).
# ---------------------------------------------------------------------------


def busy_card_uuids() -> Dict[str, str]:
    """``{gpu_uuid: pids}`` for cards with a live compute process, via
    ``nvidia-smi --query-compute-apps``. A card in this map is SKIPPED (another
    process — possibly another agent — owns it; we never fight for it). Empty on
    any error (best-effort; the measurement subprocess still fails safe)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except Exception:
        return {}
    busy: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        uuid = parts[0]
        pid = parts[1] if len(parts) > 1 else "?"
        if uuid:
            busy[uuid] = (busy.get(uuid, "") + "," + pid).strip(",")
    return busy


# ---------------------------------------------------------------------------
# The measurement (needs a GPU) — one card, in-process on the visible device.
# ---------------------------------------------------------------------------


def _norm_uuid(u) -> str:
    s = u.decode() if isinstance(u, bytes) else str(u)
    return "".join(ch for ch in s.lower() if ch in "0123456789abcdef")


def _nvml_index_for_uuid(pynvml, target_uuid: str) -> int:
    want = _norm_uuid(target_uuid)
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        if _norm_uuid(pynvml.nvmlDeviceGetUUID(h)) == want:
            return i
    raise ValueError(f"UUID {target_uuid} not found among NVML devices")


def _arch_of(pynvml, handle) -> str:
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return f"sm{major}{minor}"
    except Exception:
        return "unknown"


def _measure_active_power(sampler, dev, bench_fn, warmup_s, window_s):
    """Run ``bench_fn(dev)`` (a uneven_perf micro-bench that keeps the card busy
    AND returns its throughput) back-to-back for a warmup then a measurement
    window, and return (avg_board_watts, avg_throughput). The sampler is already
    running in the background and samples ONLY this card's NVML handle, so the
    per-card integral is this board's active draw over the window."""
    end = time.time() + warmup_s
    while time.time() < end:
        bench_fn(dev)
    t0 = time.time()
    vals = []
    end = time.time() + window_s
    while time.time() < end:
        vals.append(bench_fn(dev))
    t1 = time.time()
    _, per_card_w = sampler.integrate_per_card(t0, t1)
    watts = per_card_w[0] if per_card_w else 0.0
    thr = sum(vals) / len(vals) if vals else 0.0
    return watts, thr


def measure_card_power(
    target_uuid: str,
    *,
    sample_ms: float = _DEFAULT_SAMPLE_MS,
    idle_window_s: float = _DEFAULT_IDLE_WINDOW_S,
    active_warmup_s: float = _DEFAULT_ACTIVE_WARMUP_S,
    active_window_s: float = _DEFAULT_ACTIVE_WINDOW_S,
) -> CardPowerMeasurement:
    """Measure ONE physical card (identified by NVML ``target_uuid``) that is
    visible to torch as ``cuda:0`` — i.e. call this from a process launched with
    ``CUDA_VISIBLE_DEVICES=<target_uuid>``. Returns its MEASURED power row.

    Reuses ``energy.PowerSampler`` (per-card NVML board-power sampler) and
    ``uneven_perf._bench_membw_gbs`` / ``_bench_gemm_tflops`` (the exact decode /
    prefill micro-kernels). Frees the CUDA context on return."""
    import pynvml
    import torch

    from sglang.srt.planner.energy import PowerSampler, _nvml_name
    from sglang.srt.uneven_perf import _bench_gemm_tflops, _bench_membw_gbs

    pynvml.nvmlInit()
    nvml_idx = _nvml_index_for_uuid(pynvml, target_uuid)
    handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
    name = _nvml_name(pynvml.nvmlDeviceGetName(handle))
    arch = _arch_of(pynvml, handle)
    total_mib = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total // 2**20)
    real_uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(real_uuid, bytes):
        real_uuid = real_uuid.decode()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
    except Exception:
        driver = None

    if torch.cuda.device_count() < 1:  # pragma: no cover - hardware guard
        raise RuntimeError(
            "no CUDA device visible; launch with CUDA_VISIBLE_DEVICES=<uuid>")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    # Sample ONLY this card's NVML handle (attributed by UUID, not torch order).
    sampler = PowerSampler(gpu_indices=[nvml_idx], sample_ms=sample_ms)
    try:
        # P_idle: CUDA context alive + a small tensor resident, NO active kernel.
        resident = torch.zeros(256, dtype=torch.float32, device=dev)
        torch.cuda.synchronize(dev)
        sampler.start()
        p_idle_w = sampler.sample_now(idle_window_s)

        # P_membw (decode-relevant) then P_gemm (prefill-relevant).
        p_membw_w, membw_gbs = _measure_active_power(
            sampler, dev, _bench_membw_gbs, active_warmup_s, active_window_s)
        p_gemm_w, gemm_tflops = _measure_active_power(
            sampler, dev, _bench_gemm_tflops, active_warmup_s, active_window_s)

        del resident
        torch.cuda.empty_cache()
    finally:
        sampler.stop()
        sampler.shutdown()

    return CardPowerMeasurement(
        uuid=real_uuid,
        name=name,
        arch=arch,
        total_mib=total_mib,
        p_idle_w=round(p_idle_w, 1),
        p_membw_w=round(p_membw_w, 1),
        p_gemm_w=round(p_gemm_w, 1),
        membw_gbs=round(membw_gbs, 1),
        gemm_tflops=round(gemm_tflops, 2),
        provenance=MEASURED_PROVENANCE,
        driver=driver,
        measured_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        sample_ms=sample_ms,
        window_s=active_window_s,
    )


# ---------------------------------------------------------------------------
# Orchestration — every card, one subprocess each, isolated by UUID.
# ---------------------------------------------------------------------------


def _enumerate_cards() -> List[dict]:
    """[{uuid, name, arch, total_mib}, ...] for every NVML device (physical
    order). Import-guarded; raises only if pynvml/NVML is genuinely unavailable."""
    import pynvml

    pynvml.nvmlInit()
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            out.append({
                "uuid": uuid,
                "name": name,
                "arch": _arch_of(pynvml, h),
                "total_mib": int(pynvml.nvmlDeviceGetMemoryInfo(h).total // 2**20),
            })
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def measure_all_cards(
    *,
    only_uuids: Optional[Sequence[str]] = None,
    persist: bool = True,
    path: str = DEFAULT_POWER_PROFILE_PATH,
    subprocess_timeout_s: float = 180.0,
    python_exe: str = sys.executable,
    **measure_kwargs,
) -> PowerCalibrationResult:
    """Measure EVERY physical card (or the subset in ``only_uuids``), one at a
    time, each in its own ``CUDA_VISIBLE_DEVICES=<uuid>`` subprocess. Cards busy
    with another process are SKIPPED (never contended). Persists the MEASURED
    table by default (re-runnable — a fresh run overwrites).

    This is the reusable entry point the dashboard button calls. Returns the
    ``PowerCalibrationResult`` regardless of how many cards were reachable."""
    cards_inv = _enumerate_cards()
    if only_uuids is not None:
        want = {_norm_uuid(u) for u in only_uuids}
        cards_inv = [c for c in cards_inv if _norm_uuid(c["uuid"]) in want]

    busy = busy_card_uuids()
    busy_norm = {_norm_uuid(u): pids for u, pids in busy.items()}

    measured: List[CardPowerMeasurement] = []
    skipped: List[dict] = []
    driver = None
    for card in cards_inv:
        uuid = card["uuid"]
        if _norm_uuid(uuid) in busy_norm:
            skipped.append({
                "uuid": uuid, "name": card["name"], "arch": card["arch"],
                "reason": "busy",
                "detail": f"compute process(es) pid={busy_norm[_norm_uuid(uuid)]}",
            })
            continue
        try:
            m = _measure_one_in_subprocess(
                uuid, python_exe, subprocess_timeout_s, measure_kwargs)
            measured.append(m)
            driver = driver or m.driver
        except Exception as e:  # pragma: no cover - hardware/runtime failures
            skipped.append({
                "uuid": uuid, "name": card["name"], "arch": card["arch"],
                "reason": "error", "detail": str(e)[:400],
            })

    result = PowerCalibrationResult(
        cards=measured, skipped=skipped, driver=driver,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if persist and measured:
        save_power_profile(result, path)
    return result


def _venv_cuda_ld_path(python_exe: str) -> Optional[str]:
    """The venv's bundled NVIDIA/torch shared-lib dirs, so a fresh re-exec'd
    worker python finds libcudart before torch sets up its rpath (mirrors
    energy._venv_cuda_lib_dirs). Returned as a ``:``-joined path or None."""
    try:
        from sglang.srt.planner.energy import _venv_cuda_lib_dirs

        dirs = _venv_cuda_lib_dirs(python_exe)
        return os.pathsep.join(dirs) if dirs else None
    except Exception:
        return None


def _measure_one_in_subprocess(
    uuid: str, python_exe: str, timeout_s: float, measure_kwargs: dict
) -> CardPowerMeasurement:
    """Spawn ``python -m sglang.srt.planner.power_calibration --measure-one
    <uuid>`` with ``CUDA_VISIBLE_DEVICES=<uuid>`` so exactly this one physical
    card is visible to torch. Parses the JSON row it prints on the last line."""
    from sglang.srt.planner.energy import _default_pythonpath

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = uuid  # UUID form -> one physical GPU visible
    env.setdefault("PYTHONPATH", _default_pythonpath())
    ld = _venv_cuda_ld_path(python_exe)
    if ld:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            ld + (os.pathsep + existing if existing else ""))
    cmd = [python_exe, "-m", "sglang.srt.planner.power_calibration",
           "--measure-one", uuid]
    for k in ("sample_ms", "idle_window_s", "active_warmup_s", "active_window_s"):
        if k in measure_kwargs:
            cmd += [f"--{k.replace('_', '-')}", str(measure_kwargs[k])]
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"measure subprocess rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '')[-500:]}")
    # The row is the last non-empty stdout line (a JSON object).
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = line
            break
    if payload is None:
        raise RuntimeError(
            f"measure subprocess produced no JSON row: {proc.stdout[-500:]}")
    return CardPowerMeasurement.from_json(json.loads(payload))


# ---------------------------------------------------------------------------
# CLI: `--measure-one <uuid>` (subprocess worker) / `--all` (orchestrator).
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measure-one", metavar="UUID",
                   help="measure the single card visible as cuda:0 "
                        "(run with CUDA_VISIBLE_DEVICES=<uuid>); prints its "
                        "JSON row on stdout.")
    p.add_argument("--all", action="store_true",
                   help="orchestrate: measure every free card, persist the table.")
    p.add_argument("--path", default=DEFAULT_POWER_PROFILE_PATH)
    p.add_argument("--sample-ms", type=float, default=_DEFAULT_SAMPLE_MS)
    p.add_argument("--idle-window-s", type=float, default=_DEFAULT_IDLE_WINDOW_S)
    p.add_argument("--active-warmup-s", type=float,
                   default=_DEFAULT_ACTIVE_WARMUP_S)
    p.add_argument("--active-window-s", type=float,
                   default=_DEFAULT_ACTIVE_WINDOW_S)
    args = p.parse_args(argv)

    if args.measure_one:
        m = measure_card_power(
            args.measure_one,
            sample_ms=args.sample_ms,
            idle_window_s=args.idle_window_s,
            active_warmup_s=args.active_warmup_s,
            active_window_s=args.active_window_s,
        )
        print(json.dumps(m.to_json()))
        return 0

    # Default: orchestrate the full rig.
    result = measure_all_cards(
        persist=True, path=args.path,
        sample_ms=args.sample_ms, idle_window_s=args.idle_window_s,
        active_warmup_s=args.active_warmup_s, active_window_s=args.active_window_s,
    )
    print(json.dumps(result.to_json(), indent=1))
    for c in result.cards:
        print(f"[power] {c.name:24s} {c.arch:7s} idle={c.p_idle_w:6.1f}W "
              f"membw={c.p_membw_w:6.1f}W ({c.membw_gbs:6.1f} GB/s) "
              f"gemm={c.p_gemm_w:6.1f}W ({c.gemm_tflops:7.1f} TFLOP/s)",
              file=sys.stderr)
    for s in result.skipped:
        print(f"[power] SKIPPED {s.get('name','?')} ({s.get('reason')}): "
              f"{s.get('detail','')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
