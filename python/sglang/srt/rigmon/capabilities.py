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
"""The capability table — derived from PROBES, never from configuration.

Every capability resolves to exactly one of four states:

``active``       in use by the engine that is running now;
``available``    usable on this host, but the current run does not use it;
``unavailable``  not usable — **with a reason**;
``unknown``      the probe could not decide (typically: no engine is running,
                 so "active" is unanswerable).

The reason is the point of the whole table. Silent fallback is a recurring
defect class in this project: a transport that quietly degrades, a JIT cache
built for one arch and reused on another, an extension that fails to load and
is replaced by a slow path. Each of those cost real debugging time precisely
because the system kept working and said nothing. A capability table that
reports "not available" without saying why reproduces the failure it is meant
to expose.

The RDMA probe is the worked example. Configuration says nothing useful; only
the chain of probes does. Measured on this host: the kernel exposes two RoCE
devices under ``/sys/class/infiniband``, one of them ``ACTIVE`` at 40 Gb/sec —
and yet RDMA is unusable, because ``/dev/infiniband`` does not exist. The verbs
character devices are not mapped into the container, so libibverbs cannot open
any device; RDMA works only on the host outside it. That is a specific,
actionable sentence, and it is the sentence the probe must produce instead of
a bare "not available".
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Capability",
    "ProbeEnv",
    "CapabilityReport",
    "probe_rdma",
    "probe_p2p",
    "probe_python_module",
    "probe_all",
    "ACTIVE",
    "AVAILABLE",
    "UNAVAILABLE",
    "UNKNOWN",
]

ACTIVE = "active"
AVAILABLE = "available"
UNAVAILABLE = "unavailable"
UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class Capability:
    key: str
    label: str
    state: str
    reason: Optional[str] = None
    evidence: Dict[str, Any] = dataclasses.field(default_factory=dict)
    probe: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CapabilityReport:
    capabilities: List[Capability]
    engine_seen: bool

    def by_key(self, key: str) -> Optional[Capability]:
        for c in self.capabilities:
            if c.key == key:
                return c
        return None

    def to_json(self) -> dict:
        return {
            "engine_seen": self.engine_seen,
            "note": (
                None
                if self.engine_seen
                else "No engine reachable, so nothing can be reported as "
                "'active'; states are host capability only."
            ),
            "capabilities": [c.to_json() for c in self.capabilities],
        }


# ---------------------------------------------------------------------------
# Probe environment (injectable, so every probe is unit-testable without the
# hardware or the container it describes)
# ---------------------------------------------------------------------------


def _default_run(cmd: Sequence[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=10, check=False
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "not found"
    except Exception as e:  # timeout etc.
        return 1, f"{type(e).__name__}: {e}"


def _default_read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _default_import(name: str):
    return importlib.import_module(name)


@dataclasses.dataclass
class ProbeEnv:
    exists: Callable[[str], bool] = os.path.exists
    listdir: Callable[[str], List[str]] = os.listdir
    read_text: Callable[[str], str] = _default_read
    run: Callable[[Sequence[str]], Tuple[int, str]] = _default_run
    import_module: Callable[[str], Any] = _default_import
    env: Dict[str, str] = dataclasses.field(default_factory=lambda: dict(os.environ))
    #: ``/get_server_info`` payload of the running engine, or None.
    engine_info: Optional[Dict[str, Any]] = None
    #: A cached ``hw_profile-*.json``, which carries MEASURED pair bandwidths.
    hw_profile: Optional[Dict[str, Any]] = None

    def safe_listdir(self, path: str) -> List[str]:
        try:
            return sorted(self.listdir(path))
        except Exception:
            return []

    def safe_read(self, path: str) -> Optional[str]:
        try:
            return self.read_text(path)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# RDMA
# ---------------------------------------------------------------------------

_SYS_IB = "/sys/class/infiniband"
_SYS_UVERBS = "/sys/class/infiniband_verbs"
_DEV_IB = "/dev/infiniband"


def probe_rdma(env: ProbeEnv) -> Capability:
    """Walk the chain: kernel devices -> port link state -> verbs character
    devices -> userspace tooling. Report the FIRST link that breaks, naming it.

    The order matters. A probe that started at ``ibv_devinfo`` and stopped at
    "command not found" would hide the fact that a 40 Gb/sec link is up; a
    probe that stopped at the sysfs link state would claim RDMA works when no
    process can open a device.
    """
    evidence: Dict[str, Any] = {}

    devices = env.safe_listdir(_SYS_IB)
    evidence["kernel_devices"] = devices
    if not devices:
        return Capability(
            "rdma",
            "RDMA / InfiniBand-verbs",
            UNAVAILABLE,
            reason=(
                f"no RDMA devices under {_SYS_IB}: no RDMA-capable adapter is "
                "bound (no driver loaded, or no such hardware on this host)"
            ),
            evidence=evidence,
            probe="sysfs",
        )

    ports: List[Dict[str, Any]] = []
    for dev in devices:
        base = f"{_SYS_IB}/{dev}/ports"
        for port in env.safe_listdir(base):
            def _f(field: str) -> Optional[str]:
                v = env.safe_read(f"{base}/{port}/{field}")
                return v.strip() if v else None

            state = _f("state") or ""
            ports.append(
                {
                    "device": dev,
                    "port": port,
                    "state": state,
                    "phys_state": _f("phys_state"),
                    "rate": _f("rate"),
                    "link_layer": _f("link_layer"),
                    "up": "ACTIVE" in state.upper(),
                }
            )
    evidence["ports"] = ports
    up = [p for p in ports if p["up"]]
    evidence["ports_up"] = len(up)

    # The container trap: kernel devices present, but the verbs character
    # devices are not mapped in, so no userspace process can open one.
    dev_ib = env.exists(_DEV_IB)
    evidence["dev_infiniband"] = dev_ib
    uverbs = env.safe_listdir(_SYS_UVERBS)
    evidence["uverbs_class"] = uverbs
    if not dev_ib:
        detail = (
            f"{len(devices)} RDMA device(s) visible to the kernel"
            + (
                f" and {len(up)} port(s) up ("
                + ", ".join(f"{p['device']}:{p['port']} {p['rate']}" for p in up)
                + ")"
                if up
                else " but no port is up"
            )
        )
        return Capability(
            "rdma",
            "RDMA / InfiniBand-verbs",
            UNAVAILABLE,
            reason=(
                f"{detail}, but {_DEV_IB} does not exist"
                + (
                    f" although {_SYS_UVERBS} lists {', '.join(uverbs)}"
                    if uverbs
                    else ""
                )
                + ". The verbs character devices are not mapped into this "
                "container, so libibverbs cannot open a device. RDMA is usable "
                "only outside the container (on the hypervisor host), or after "
                "the device nodes are passed through."
            ),
            evidence=evidence,
            probe="sysfs + /dev",
        )

    if not up:
        return Capability(
            "rdma",
            "RDMA / InfiniBand-verbs",
            UNAVAILABLE,
            reason=(
                f"{len(devices)} device(s) present and {_DEV_IB} exists, but no "
                "port is ACTIVE: "
                + "; ".join(
                    f"{p['device']}:{p['port']} state={p['state']} "
                    f"phys={p['phys_state']}"
                    for p in ports
                )
                + ". The link is down — cabling, switch port, or the interface "
                "is administratively disabled."
            ),
            evidence=evidence,
            probe="sysfs + /dev",
        )

    # Userspace tooling: informative, not decisive. Its absence does not make
    # RDMA unusable, so it must not flip the state.
    rc, out = env.run(["ibv_devinfo"])
    evidence["ibv_devinfo_rc"] = rc
    if rc == 0:
        evidence["ibv_devinfo"] = out[:4000]
    else:
        evidence["ibv_devinfo_note"] = (
            "ibv_devinfo not installed or failed; this does not disable RDMA, "
            "it only removes a cross-check (package: ibverbs-utils / rdma-core)"
        )

    active = _engine_uses_rdma(env)
    return Capability(
        "rdma",
        "RDMA / InfiniBand-verbs",
        ACTIVE if active else AVAILABLE,
        reason=(
            None
            if active
            else "device open and link up, but the running configuration does "
            "not select an RDMA transport"
        ),
        evidence=evidence,
        probe="sysfs + /dev + ibv_devinfo",
    )


def _engine_uses_rdma(env: ProbeEnv) -> bool:
    info = env.engine_info or {}
    for key in ("disaggregation_transfer_backend", "transport", "ccl_transport"):
        v = str(info.get(key) or "").lower()
        if any(t in v for t in ("rdma", "roce", "ib", "ucx", "nixl", "mooncake")):
            return True
    for key, val in env.env.items():
        if key.startswith(("UCX_TLS", "NCCL_IB_")) and val:
            if key == "NCCL_IB_DISABLE" and val.strip() in ("1", "true", "TRUE"):
                return False
            return True
    return False


# ---------------------------------------------------------------------------
# GPU-to-GPU peer access
# ---------------------------------------------------------------------------


def probe_p2p(env: ProbeEnv) -> Capability:
    """Peer-to-peer between cards.

    The MEASURED probe result outranks the topology matrix: this rig reports
    a topology that suggests peer access, while the measured pair bandwidths
    are all in the single-digit GB/s range, i.e. host-staged. Where a
    measurement exists, it decides.
    """
    evidence: Dict[str, Any] = {}
    links = ((env.hw_profile or {}).get("links") or {})
    pair_gbs = {
        k: v.get("p2p_gbs")
        for k, v in links.items()
        if k != "__group__" and isinstance(v, dict)
    }
    if pair_gbs:
        evidence["measured_pair_gbs"] = pair_gbs
        best = max((v for v in pair_gbs.values() if v is not None), default=None)
        evidence["best_pair_gbs"] = best
        if best is not None and best < 20.0:
            return Capability(
                "p2p",
                "GPU peer-to-peer",
                UNAVAILABLE,
                reason=(
                    f"measured pair bandwidth peaks at {best:.1f} GB/s, which is "
                    "host-staged PCIe traffic, not peer access. Collectives are "
                    "paying a bounce through host memory."
                ),
                evidence=evidence,
                probe="hardware probe (measured)",
            )
        return Capability(
            "p2p",
            "GPU peer-to-peer",
            AVAILABLE,
            evidence=evidence,
            probe="hardware probe (measured)",
        )

    rc, out = env.run(["nvidia-smi", "topo", "-p2p", "r"])
    evidence["topo_rc"] = rc
    if rc != 0:
        return Capability(
            "p2p",
            "GPU peer-to-peer",
            UNKNOWN,
            reason=(
                "no measured pair bandwidths (run the hardware probe) and "
                f"`nvidia-smi topo -p2p r` failed: {out.strip()[:200]}"
            ),
            evidence=evidence,
            probe="nvidia-smi topo",
        )
    evidence["topo"] = out[:2000]
    if "OK" in out:
        return Capability(
            "p2p",
            "GPU peer-to-peer",
            AVAILABLE,
            reason=(
                "topology reports peer access, but no pair bandwidth has been "
                "measured — the topology matrix has been wrong on this class of "
                "board before. Run the probe to confirm."
            ),
            evidence=evidence,
            probe="nvidia-smi topo",
        )
    return Capability(
        "p2p",
        "GPU peer-to-peer",
        UNAVAILABLE,
        reason=(
            "no pair reports peer access in `nvidia-smi topo -p2p r` (typical "
            "for consumer boards behind a chipset)"
        ),
        evidence=evidence,
        probe="nvidia-smi topo",
    )


# ---------------------------------------------------------------------------
# Python-module-backed capabilities
# ---------------------------------------------------------------------------


def probe_python_module(
    env: ProbeEnv,
    key: str,
    label: str,
    module: str,
    active_when: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Capability:
    """Import-probe a backend package. An import that RAISES carries the
    exception text as the reason — that is where "built for sm86, running on
    sm120" surfaces, and it is exactly the message that must not be swallowed.
    """
    evidence: Dict[str, Any] = {"module": module}
    try:
        mod = env.import_module(module)
        evidence["version"] = str(getattr(mod, "__version__", "") or "") or None
        evidence["path"] = str(getattr(mod, "__file__", "") or "") or None
    except Exception as e:
        return Capability(
            key,
            label,
            UNAVAILABLE,
            reason=f"import {module} failed: {type(e).__name__}: {e}",
            evidence=evidence,
            probe="python import",
        )
    info = env.engine_info
    if info is None:
        return Capability(
            key,
            label,
            AVAILABLE,
            reason="installed; no engine reachable, so use cannot be confirmed",
            evidence=evidence,
            probe="python import",
        )
    if active_when is not None and active_when(info):
        return Capability(key, label, ACTIVE, evidence=evidence, probe="python import")
    return Capability(
        key,
        label,
        AVAILABLE,
        reason="installed, but the running configuration does not select it",
        evidence=evidence,
        probe="python import",
    )


# ---------------------------------------------------------------------------
# Engine-configuration-backed capabilities
# ---------------------------------------------------------------------------


def _engine_flag(
    env: ProbeEnv,
    key: str,
    label: str,
    active_when: Callable[[Dict[str, Any]], bool],
    unavailable_when: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> Capability:
    info = env.engine_info
    if info is None:
        return Capability(
            key,
            label,
            UNKNOWN,
            reason="no engine reachable; this is a per-run setting, not a host property",
            probe="engine /get_server_info",
        )
    if unavailable_when is not None:
        why = unavailable_when(info)
        if why:
            return Capability(
                key, label, UNAVAILABLE, reason=why, probe="engine /get_server_info"
            )
    if active_when(info):
        return Capability(key, label, ACTIVE, probe="engine /get_server_info")
    return Capability(
        key,
        label,
        AVAILABLE,
        reason="supported by this build, off in the running configuration",
        probe="engine /get_server_info",
    )


def _truthy(v) -> bool:
    return bool(v) and str(v).lower() not in ("0", "false", "none", "")


def probe_all(env: Optional[ProbeEnv] = None) -> CapabilityReport:
    """The full table."""
    env = env or ProbeEnv()
    caps: List[Capability] = [probe_rdma(env), probe_p2p(env)]

    caps.append(
        probe_python_module(
            env,
            "flashinfer",
            "FlashInfer attention backend",
            "flashinfer",
            active_when=lambda i: "flashinfer"
            in str(i.get("attention_backend") or "").lower(),
        )
    )
    caps.append(
        probe_python_module(
            env,
            "sgl_kernel",
            "sgl-kernel (Marlin / quantised GEMM)",
            "sgl_kernel",
            active_when=lambda i: "marlin"
            in str(i.get("quantization") or "").lower(),
        )
    )
    caps.append(
        _engine_flag(
            env,
            "fp8_kv",
            "fp8 KV cache",
            active_when=lambda i: "fp8" in str(i.get("kv_cache_dtype") or "").lower(),
        )
    )
    caps.append(
        _engine_flag(
            env,
            "cuda_graphs",
            "CUDA graphs",
            active_when=lambda i: not _truthy(i.get("disable_cuda_graph")),
            unavailable_when=lambda i: (
                "disabled in this run by --disable-cuda-graph"
                if _truthy(i.get("disable_cuda_graph"))
                else None
            ),
        )
    )
    caps.append(
        _engine_flag(
            env,
            "speculation",
            "speculative decoding",
            active_when=lambda i: _truthy(i.get("speculative_algorithm"))
            or _truthy(i.get("speculative_draft_model_path")),
        )
    )
    caps.append(
        _engine_flag(
            env,
            "dcp",
            "decode context parallelism",
            active_when=lambda i: (i.get("dcp_size") or 1) > 1,
        )
    )
    caps.append(
        _engine_flag(
            env,
            "uneven_tp",
            "uneven tensor parallelism",
            active_when=lambda i: _truthy(i.get("rank_tp_ratio"))
            or _truthy(i.get("rank_gpu_id")),
        )
    )
    return CapabilityReport(capabilities=caps, engine_seen=env.engine_info is not None)
