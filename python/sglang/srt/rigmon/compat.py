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
"""The compatibility gate: run BEFORE two nodes are joined into one rig.

Discovery and health checking already exist in the router
(``experimental/sgl-router/src/discovery``, ``src/health``), so neither is
rebuilt here. What is missing — and what decides between "one click" and "one
click followed by twenty minutes of debugging" — is the gate that runs before
the two sides are allowed to work together:

* the same fork commit on both sides, and neither working tree dirty;
* compatible CUDA / driver and torch versions;
* an architecture list that covers BOTH sides' cards;
* the model present on both sides with the same fingerprint.

Each of those is a failure this project has actually hit: a kernel built for
one architecture and reused on the other, a cache key without an architecture
component, a checkpoint that differed between machines. The point of checking
them at join time is that all of them otherwise surface much later, as a crash
in a worker or, worse, as a silent quality difference.

Verdicts are three-valued. ``block`` means joining will fail or produce wrong
results. ``warn`` means it will probably work but something is unequal and
should be visible. ``ok`` speaks for itself. Nothing here is advisory-only by
accident: the severity is a property of the check, not of the caller's mood.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from sglang.srt.rigmon.provenance import git_commit, model_fingerprint

__all__ = [
    "NodeIdentity",
    "CompatCheck",
    "CompatReport",
    "local_identity",
    "check_compatibility",
    "BLOCK",
    "WARN",
    "OK",
]

BLOCK = "block"
WARN = "warn"
OK = "ok"


@dataclasses.dataclass
class NodeIdentity:
    """What a node presents at the gate."""

    node_id: str
    commit: Optional[str] = None
    branch: Optional[str] = None
    dirty: Optional[bool] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    #: Compute capabilities of this node's cards, e.g. ["sm86", "sm120"].
    gpu_archs: List[str] = dataclasses.field(default_factory=list)
    #: Architectures the installed extensions were BUILT for.
    built_archs: List[str] = dataclasses.field(default_factory=list)
    gpu_names: List[str] = dataclasses.field(default_factory=list)
    #: model path -> fingerprint dict.
    models: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "NodeIdentity":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _torch_info() -> Dict[str, Any]:
    """Versions and architecture lists, read without initialising CUDA."""
    out: Dict[str, Any] = {}
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        # get_arch_list() reports what the build supports; it does not create
        # a context, unlike querying a device.
        archs = []
        for a in getattr(torch.cuda, "get_arch_list", lambda: [])():
            m = re.search(r"(\d+)", str(a))
            if m:
                archs.append(f"sm{m.group(1)}")
        out["built_archs"] = sorted(set(archs))
    except Exception:
        pass
    return out


def _nvml_archs() -> Dict[str, List[str]]:
    """Card names and compute capabilities via NVML (read-only)."""
    names: List[str] = []
    archs: List[str] = []
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                n = pynvml.nvmlDeviceGetName(h)
                names.append(n.decode() if isinstance(n, bytes) else str(n))
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                    archs.append(f"sm{major}{minor}")
                except Exception:
                    pass
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return {"gpu_names": names, "gpu_archs": sorted(set(archs))}


def _driver_version() -> Optional[str]:
    try:
        with open("/proc/driver/nvidia/version") as f:
            for tok in f.read().split():
                if tok[:3].isdigit() and "." in tok:
                    return tok
    except OSError:
        pass
    return None


def local_identity(
    node_id: str,
    model_paths: Sequence[str] = (),
    repo_dir: Optional[str] = None,
) -> NodeIdentity:
    """Assemble this node's identity. Read-only; creates no CUDA context."""
    git = git_commit(repo_dir)
    ti = _torch_info()
    nv = _nvml_archs()
    return NodeIdentity(
        node_id=node_id,
        commit=git.get("commit"),
        branch=git.get("branch"),
        dirty=git.get("dirty"),
        torch_version=ti.get("torch_version"),
        cuda_version=ti.get("cuda_version"),
        driver_version=_driver_version(),
        gpu_archs=nv["gpu_archs"],
        built_archs=ti.get("built_archs", []),
        gpu_names=nv["gpu_names"],
        models={p: model_fingerprint(p) for p in model_paths},
    )


@dataclasses.dataclass
class CompatCheck:
    key: str
    label: str
    verdict: str
    detail: str
    remedy: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CompatReport:
    local: str
    remote: str
    checks: List[CompatCheck]

    @property
    def blocked(self) -> bool:
        return any(c.verdict == BLOCK for c in self.checks)

    @property
    def warnings(self) -> List[CompatCheck]:
        return [c for c in self.checks if c.verdict == WARN]

    def to_json(self) -> dict:
        return {
            "local": self.local,
            "remote": self.remote,
            "blocked": self.blocked,
            "checks": [c.to_json() for c in self.checks],
        }


def check_compatibility(
    local: NodeIdentity, remote: NodeIdentity
) -> CompatReport:
    """Run the gate. Order is deliberate: the cheapest and most common
    mismatches first, so the report leads with what usually went wrong."""
    checks: List[CompatCheck] = []

    # -- fork commit --------------------------------------------------------
    if not local.commit or not remote.commit:
        checks.append(
            CompatCheck(
                "commit", "fork commit", WARN,
                "at least one side did not report a commit",
                "run both collectors from a git checkout so the build can be "
                "identified",
            )
        )
    elif local.commit != remote.commit:
        checks.append(
            CompatCheck(
                "commit", "fork commit", BLOCK,
                f"local {local.commit[:12]} ({local.branch}) vs remote "
                f"{remote.commit[:12]} ({remote.branch})",
                "check out the same commit on both nodes. Cross-rig work "
                "shares wire formats and split arithmetic, and neither is "
                "stable across commits.",
            )
        )
    else:
        checks.append(
            CompatCheck("commit", "fork commit", OK, f"both on {local.commit[:12]}")
        )

    dirty = [n.node_id for n in (local, remote) if n.dirty]
    if dirty:
        checks.append(
            CompatCheck(
                "dirty_tree", "working tree", WARN,
                f"uncommitted changes on {', '.join(dirty)}",
                "the commit no longer identifies the code that will run; "
                "commit or stash before relying on a comparison",
            )
        )

    # -- architecture coverage ---------------------------------------------
    # The recurring failure: an extension built for one card's architecture,
    # reused on the other, surfacing as cudaErrorNoKernelImageForDevice deep
    # into a run rather than at start-up.
    all_archs = sorted(set(local.gpu_archs) | set(remote.gpu_archs))
    for node in (local, remote):
        if not node.built_archs or not all_archs:
            continue
        missing = [a for a in all_archs if a not in node.built_archs]
        if missing:
            checks.append(
                CompatCheck(
                    f"arch_coverage:{node.node_id}",
                    f"architecture coverage on {node.node_id}",
                    BLOCK,
                    f"built for {', '.join(node.built_archs)} but the joined "
                    f"rig contains {', '.join(all_archs)}; missing "
                    f"{', '.join(missing)}",
                    "rebuild with every architecture of the JOINED rig in "
                    "TORCH_CUDA_ARCH_LIST, and clear the JIT cache — a cache "
                    "key without an architecture component is how this failure "
                    "recurs.",
                )
            )
    if all_archs and all(
        n.built_archs and not [a for a in all_archs if a not in n.built_archs]
        for n in (local, remote)
    ):
        checks.append(
            CompatCheck(
                "arch_coverage", "architecture coverage", OK,
                f"both builds cover {', '.join(all_archs)}",
            )
        )

    # -- toolchain ----------------------------------------------------------
    for key, label, lv, rv, sev in (
        ("torch", "torch version", local.torch_version, remote.torch_version, BLOCK),
        ("cuda", "CUDA version", local.cuda_version, remote.cuda_version, WARN),
        ("driver", "driver version", local.driver_version, remote.driver_version, WARN),
    ):
        if lv is None or rv is None:
            continue
        if lv != rv:
            checks.append(
                CompatCheck(
                    key, label, sev, f"local {lv} vs remote {rv}",
                    "align the two sides"
                    if sev == BLOCK
                    else "usually tolerable; recorded so a later difference is "
                    "not mistaken for a code change",
                )
            )
        else:
            checks.append(CompatCheck(key, label, OK, f"both {lv}"))

    # -- models -------------------------------------------------------------
    shared = set(local.models) & set(remote.models)
    only_local = sorted(set(local.models) - shared)
    only_remote = sorted(set(remote.models) - shared)
    for path in sorted(shared):
        lf, rf = local.models[path], remote.models[path]
        if not lf.get("exists") or not rf.get("exists"):
            missing_on = local.node_id if not lf.get("exists") else remote.node_id
            checks.append(
                CompatCheck(
                    f"model:{path}", f"model {os.path.basename(path)}", BLOCK,
                    f"not present on {missing_on}",
                    f"make {path} available on both nodes",
                )
            )
        elif lf.get("fingerprint") != rf.get("fingerprint"):
            checks.append(
                CompatCheck(
                    f"model:{path}", f"model {os.path.basename(path)}", BLOCK,
                    f"fingerprints differ ({lf.get('fingerprint')} vs "
                    f"{rf.get('fingerprint')})",
                    "the two sides hold different checkpoints. Cross-rig work "
                    "hands intermediate state between them, so differing "
                    "weights produce a quality seam rather than an error.",
                )
            )
        else:
            checks.append(
                CompatCheck(
                    f"model:{path}", f"model {os.path.basename(path)}", OK,
                    f"identical on both ({lf.get('fingerprint')})",
                )
            )
    if only_local or only_remote:
        checks.append(
            CompatCheck(
                "model_sets", "declared model sets", WARN,
                f"only on {local.node_id}: {only_local or 'none'}; only on "
                f"{remote.node_id}: {only_remote or 'none'}",
                "models present on one side only cannot serve a joined run",
            )
        )
    if not local.models and not remote.models:
        checks.append(
            CompatCheck(
                "model_sets", "declared model sets", WARN,
                "neither node declared a model",
                "pass the model path to both collectors so the gate can "
                "compare checkpoints",
            )
        )

    return CompatReport(local=local.node_id, remote=remote.node_id, checks=checks)
