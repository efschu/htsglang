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
"""Facilities: what this host can MEASURE and CONTROL, and what it would take.

The capability table in :mod:`~sglang.srt.rigmon.capabilities` answers "is this
available, and if not, why". This module is the next stage, and the difference
is the third answer: **what would have to change for it to work.**

Every measurement and every control declares its requirements. When one is
unmet the caller gets three things, not one:

1. that it does not work,
2. why it does not work,
3. what this container (or host) would need for it to work — device nodes,
   capabilities, privileged mode, host networking — or that it is impossible
   from a container at all.

The rule exists because the negative case is where the time goes. Measured on
this host: the process runs in an **LXC guest** with a FULL capability set
(``CapEff: 000001ffffffffff``), and driver 595.58.03 still refuses ``-pm``,
``-lgc``, ``-lmc`` and ``-pl``. A UI that inferred "root plus all capabilities,
therefore clock control works" would be wrong, and one that showed a greyed-out
button with no explanation would send the user hunting for a permission problem
that does not exist. The honest sentence is "clock control must come from the
Proxmox host; an LXC guest cannot reach the driver's control path, and no
capability grant changes that".

Controls whose requirements are unmet stay **visible and disabled**. Hiding
them turns a known limitation into a missing feature.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "Requirement",
    "Facility",
    "HostEnvironment",
    "detect_host_environment",
    "facilities",
    "MEASURE",
    "CONTROL",
]

MEASURE = "measure"
CONTROL = "control"


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Requirement:
    """One precondition, with the fix attached.

    ``remedy`` is the sentence the UI shows next to a disabled control. When a
    requirement cannot be met from a container at any configuration,
    ``possible_in_container`` is False and the remedy says so plainly instead
    of suggesting a setting that will not help.
    """

    key: str
    label: str
    satisfied: bool
    detail: str = ""
    remedy: Optional[str] = None
    possible_in_container: bool = True

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Facility:
    """A measurement or a control, with its requirement chain."""

    key: str
    label: str
    kind: str  # MEASURE | CONTROL
    requirements: List[Requirement]
    #: Why it matters — shown so a disabled control is still informative.
    purpose: str = ""

    @property
    def available(self) -> bool:
        return all(r.satisfied for r in self.requirements)

    @property
    def blockers(self) -> List[Requirement]:
        return [r for r in self.requirements if not r.satisfied]

    @property
    def reason(self) -> Optional[str]:
        if self.available:
            return None
        return "; ".join(f"{r.label}: {r.detail}" for r in self.blockers)

    @property
    def remedy(self) -> List[str]:
        out = []
        for r in self.blockers:
            if r.remedy:
                out.append(r.remedy)
        return out

    @property
    def impossible_in_container(self) -> bool:
        return any(not r.possible_in_container for r in self.blockers)

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "purpose": self.purpose,
            "available": self.available,
            "reason": self.reason,
            "remedy": self.remedy,
            "impossible_in_container": self.impossible_in_container,
            # Controls stay VISIBLE and disabled. Hiding a known limitation
            # makes it look like a missing feature.
            "ui": "enabled" if self.available else "visible_disabled",
            "requirements": [r.to_json() for r in self.requirements],
        }


# ---------------------------------------------------------------------------
# Host environment
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HostEnvironment:
    """What kind of place this process is running in."""

    #: "" (bare metal / VM), "lxc", "docker", "podman", "kubernetes", ...
    container: str = ""
    virt: str = ""
    is_root: bool = False
    cap_eff: Optional[int] = None
    #: True when every capability bit in the bounding set is held.
    full_caps: bool = False
    driver_version: Optional[str] = None

    @property
    def in_container(self) -> bool:
        return bool(self.container)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["cap_eff"] = hex(self.cap_eff) if self.cap_eff is not None else None
        d["in_container"] = self.in_container
        return d


def detect_host_environment(
    read_text: Optional[Callable[[str], str]] = None,
    exists: Optional[Callable[[str], bool]] = None,
    run: Optional[Callable[[Sequence[str]], str]] = None,
    getuid: Optional[Callable[[], int]] = None,
) -> HostEnvironment:
    """Detect container kind, privilege and driver version, all read-only."""

    def _read(path: str) -> Optional[str]:
        try:
            return (read_text or _default_read)(path)
        except Exception:
            return None

    _exists = exists or os.path.exists

    def _run(cmd: Sequence[str]) -> Optional[str]:
        try:
            if run is not None:
                return run(cmd)
            return subprocess.check_output(
                list(cmd), text=True, timeout=10, stderr=subprocess.DEVNULL
            )
        except Exception:
            return None

    env = HostEnvironment()
    env.is_root = (getuid or os.getuid)() == 0

    init_env = _read("/proc/1/environ") or ""
    for entry in init_env.replace("\0", "\n").splitlines():
        if entry.startswith("container="):
            env.container = entry.split("=", 1)[1].strip()
            break
    if not env.container:
        if _exists("/.dockerenv"):
            env.container = "docker"
        elif _exists("/run/.containerenv"):
            env.container = "podman"
    virt = _run(["systemd-detect-virt"])
    if virt:
        env.virt = virt.strip()
        if not env.container and env.virt in ("lxc", "lxc-libvirt", "docker", "podman"):
            env.container = env.virt

    status = _read("/proc/self/status") or ""
    cap_eff = cap_bnd = None
    for line in status.splitlines():
        if line.startswith("CapEff:"):
            try:
                cap_eff = int(line.split()[1], 16)
            except (IndexError, ValueError):
                pass
        elif line.startswith("CapBnd:"):
            try:
                cap_bnd = int(line.split()[1], 16)
            except (IndexError, ValueError):
                pass
    env.cap_eff = cap_eff
    env.full_caps = bool(cap_eff and cap_bnd and cap_eff == cap_bnd)

    ver = _read("/proc/driver/nvidia/version")
    if ver:
        for tok in ver.split():
            if tok[:3].isdigit() and "." in tok:
                env.driver_version = tok
                break
    return env


def _default_read(path: str) -> str:
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Requirement builders
# ---------------------------------------------------------------------------


def _req_device_node(path: str, exists: Callable[[str], bool]) -> Requirement:
    ok = exists(path)
    return Requirement(
        key=f"device_node:{path}",
        label=f"device node {path}",
        satisfied=ok,
        detail="present" if ok else "absent",
        remedy=(
            None
            if ok
            else f"pass {path} into the container (LXC: a `lxc.mount.entry` or "
            f"`dev0` entry for {path}; Docker: `--device {path}`), or run the "
            "collector on the host"
        ),
    )


def _req_binary(name: str) -> Requirement:
    path = shutil.which(name)
    return Requirement(
        key=f"binary:{name}",
        label=f"`{name}` on PATH",
        satisfied=bool(path),
        detail=path or "not installed",
        remedy=None if path else f"install the package providing `{name}`",
    )


def _req_gpu_control(env: HostEnvironment) -> Requirement:
    """Driver-level clock / power control.

    Deliberately NOT inferred from the capability set. Measured here: full
    CapEff and root, and the driver still refuses. Container kind is the
    predictor, privilege is not.
    """
    if env.container:
        return Requirement(
            key="gpu_control_path",
            label="driver clock/power control path",
            satisfied=False,
            detail=(
                f"running inside a {env.container} guest"
                + (
                    f" with a full capability set ({hex(env.cap_eff)})"
                    if env.full_caps and env.cap_eff
                    else ""
                )
                + "; driver "
                + (env.driver_version or "(version unknown)")
                + " refuses -pm / -lgc / -lmc / -pl here regardless of privilege"
            ),
            remedy=(
                "run the collector on the hypervisor host. Clock and power "
                "control is not reachable from a container: it is not a "
                "capability or device-node question, so neither `privileged` "
                "nor extra capabilities enable it."
            ),
            possible_in_container=False,
        )
    if not env.is_root:
        return Requirement(
            key="gpu_control_path",
            label="driver clock/power control path",
            satisfied=False,
            detail="not root",
            remedy="run as root (nvidia-smi control operations require it)",
        )
    return Requirement(
        key="gpu_control_path",
        label="driver clock/power control path",
        satisfied=True,
        detail="on the host, as root",
    )


def _req_gpm(gpm_supported: Optional[bool], reason: Optional[str]) -> Requirement:
    if gpm_supported:
        return Requirement(
            key="gpm", label="NVML GPM profiling counters", satisfied=True,
            detail="supported",
        )
    return Requirement(
        key="gpm",
        label="NVML GPM / DCGM profiling counters",
        satisfied=False,
        detail=reason or "not supported on this device",
        remedy=(
            "GPM is a datacentre-GPU feature (Hopper and later); on consumer "
            "cards no container setting enables it. DCGM can supply the same "
            "counters on supported hardware and needs /dev/nvidia-caps plus "
            "the nvidia-dcgm service."
        ),
        possible_in_container=False,
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def facilities(
    env: Optional[HostEnvironment] = None,
    exists: Optional[Callable[[str], bool]] = None,
    gpm_supported: Optional[bool] = None,
    gpm_reason: Optional[str] = None,
) -> List[Facility]:
    """Everything the dashboard may want to measure or set, with its chain."""
    env = env or detect_host_environment()
    _exists = exists or os.path.exists
    ctl = _req_gpu_control(env)
    smi = _req_binary("nvidia-smi")

    out = [
        Facility(
            key="power_target",
            label="GPU power target",
            kind=CONTROL,
            purpose=(
                "Sweeping the power limit is the cheapest energy-per-token "
                "lever and the only way to separate a thermal ceiling from a "
                "power ceiling."
            ),
            requirements=[smi, ctl],
        ),
        Facility(
            key="clock_lock",
            label="fixed SM / memory clocks",
            kind=CONTROL,
            purpose=(
                "Without pinned clocks a split comparison measures the P-state "
                "rather than the split; the flat-decode finding is explicitly "
                "provisional until clocks are nailed down."
            ),
            requirements=[smi, ctl],
        ),
        Facility(
            key="persistence_mode",
            label="driver persistence mode",
            kind=CONTROL,
            purpose="Removes driver-load latency from short measurements.",
            requirements=[smi, ctl],
        ),
        Facility(
            key="profiling_counters",
            label="SM / tensor / DRAM activity counters",
            kind=MEASURE,
            purpose=(
                "The per-rank work share, work per watt and roofline position "
                "all rest on these; without them the display falls back to "
                "coarse utilization and says so."
            ),
            requirements=[_req_gpm(gpm_supported, gpm_reason)],
        ),
        Facility(
            key="rdma",
            label="RDMA verbs access",
            kind=MEASURE,
            purpose="Cross-rig transport, and the pair matrix that selects it.",
            requirements=[_req_device_node("/dev/infiniband", _exists)],
        ),
        Facility(
            key="ram_timings",
            label="host RAM clock and timings",
            kind=MEASURE,
            purpose=(
                "The spill/offload path is host-memory bound, so the RAM clock "
                "is a first-order term in any spill measurement."
            ),
            requirements=[
                _req_binary("dmidecode"),
                Requirement(
                    key="dev_mem",
                    label="/dev/mem (SMBIOS read)",
                    satisfied=_exists("/dev/mem"),
                    detail="present" if _exists("/dev/mem") else "absent",
                    remedy=(
                        "dmidecode reads SMBIOS through /dev/mem; grant the "
                        "container CAP_SYS_RAWIO and the /dev/mem node, or read "
                        "the value once on the host and enter it manually. RAM "
                        "clock cannot be CHANGED from software at all — it is a "
                        "firmware setting."
                    ),
                ),
            ],
        ),
        Facility(
            key="sensors",
            label="board sensors (hwmon)",
            kind=MEASURE,
            purpose="Ambient and VRM temperatures, for thermal attribution.",
            requirements=[
                Requirement(
                    key="hwmon",
                    label="/sys/class/hwmon",
                    satisfied=_exists("/sys/class/hwmon"),
                    detail=(
                        "present" if _exists("/sys/class/hwmon") else "not mounted"
                    ),
                    remedy="mount /sys read-only into the container",
                )
            ],
        ),
    ]
    return out
