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
"""Run provenance and history comparison.

A number without its provenance cannot be compared with another number. This
module records, per run: the exact flag set, the fork commit, the model
identity, the probe result in force, and the machine STATE the run happened
in.

The state part is what makes a history readable. Two runs of the same
configuration can differ because a card was thermally limited during one of
them; without that annotation the difference reads as a regression and someone
goes looking for a code change that does not exist. A card sitting at 88 C with
a thermal slowdown active and 1695 instead of 1905 MHz has been observed here,
and a comparison that ignored it would have been actively misleading.

So :func:`compare_runs` never reports a delta on its own. It reports the delta
together with whether the runs were comparable, and it refuses to call a
difference a regression when the state differed or when the difference sits
inside the measured noise floor.

Secrets never enter a record: the environment is filtered to a known list of
non-sensitive keys, so a token in the environment cannot reach a client.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "RunProvenance",
    "RunRecord",
    "capture_provenance",
    "state_summary",
    "compare_runs",
    "git_commit",
    "model_fingerprint",
    "SAFE_ENV_KEYS",
]


#: Environment variables recorded verbatim. An allow-list, not a deny-list: a
#: deny-list leaks the variable someone adds next week.
SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MAX_CTAS",
    "NCCL_NVLS_ENABLE",
    "UCX_TLS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "SGLANG_UNEVEN_MLP_VECTOR",
    "SGLANG_UNEVEN_MOE_VECTOR",
    "SGLANG_UNEVEN_VOCAB_VECTOR",
    "SGLANG_UNEVEN_TOKEN_VECTOR",
    "SGLANG_MEASURED_KV_BUDGET",
    "SGLANG_PERF_REPROBE",
    "TORCH_CUDA_ARCH_LIST",
)


def git_commit(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Commit and dirty flag. A dirty tree means the commit does not identify
    the code that ran, and a comparison across runs must say so."""
    repo_dir = repo_dir or os.getcwd()
    out: Dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            timeout=10,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"], text=True, timeout=20
        )
        out["dirty"] = bool(status.strip())
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def model_fingerprint(path: str, max_files: int = 64) -> Dict[str, Any]:
    """Identify a checkpoint without reading gigabytes.

    Size plus mtime plus name for each weight file, hashed. It catches a
    swapped or re-quantised checkpoint, which is what a comparison needs; it is
    not a content hash and does not claim to be.
    """
    out: Dict[str, Any] = {"path": path, "exists": os.path.exists(path)}
    if not out["exists"]:
        return out
    entries = []
    try:
        if os.path.isfile(path):
            st = os.stat(path)
            entries.append((os.path.basename(path), st.st_size, int(st.st_mtime)))
        else:
            for name in sorted(os.listdir(path))[:max_files]:
                if not name.endswith((".safetensors", ".bin", ".gguf", ".json")):
                    continue
                st = os.stat(os.path.join(path, name))
                entries.append((name, st.st_size, int(st.st_mtime)))
    except OSError as e:
        out["error"] = str(e)
        return out
    h = hashlib.sha256()
    for name, size, mtime in entries:
        h.update(f"{name}:{size}:{mtime}\n".encode())
    out["files"] = len(entries)
    out["total_bytes"] = sum(e[1] for e in entries)
    out["fingerprint"] = h.hexdigest()[:16]
    out["method"] = "name+size+mtime digest, not a content hash"
    return out


@dataclasses.dataclass
class RunProvenance:
    """Everything needed to repeat a run and to judge a comparison."""

    run_id: str
    created: float
    flags: List[str] = dataclasses.field(default_factory=list)
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    git: Dict[str, Any] = dataclasses.field(default_factory=dict)
    model: Dict[str, Any] = dataclasses.field(default_factory=dict)
    draft_model: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Which probe result the plan was derived from, and how old it was.
    probe: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Which persisted KV budget was in force.
    kv_budget: Dict[str, Any] = dataclasses.field(default_factory=dict)
    hardware: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    driver: Optional[str] = None
    node_id: Optional[str] = None
    notes: List[str] = dataclasses.field(default_factory=list)

    def command_line(self, program: str = "python -m sglang.launch_server") -> str:
        """The copyable start line. The dry run the planner offers is this
        string: many runs here are started by hand."""
        return " ".join([program] + list(self.flags))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def repeatable(self) -> Dict[str, Any]:
        """Can this run be repeated exactly, and if not, why not?"""
        blockers = []
        if self.git.get("dirty"):
            blockers.append(
                "the working tree was dirty, so the commit does not identify "
                "the code that ran"
            )
        if not self.git.get("commit"):
            blockers.append("no commit recorded")
        if not self.model.get("fingerprint"):
            blockers.append("model not fingerprinted")
        if self.kv_budget.get("present") and not self.kv_budget.get("config_hash"):
            blockers.append(
                "a persisted KV budget was in force but was not identified; "
                "max_total_num_tokens may not reproduce"
            )
        return {"repeatable": not blockers, "blockers": blockers}


def capture_provenance(
    flags: Sequence[str],
    model_path: str = "",
    draft_model_path: str = "",
    probe: Optional[dict] = None,
    kv_budget: Optional[Any] = None,
    cards: Optional[Sequence[dict]] = None,
    node_id: Optional[str] = None,
    repo_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    now: Optional[float] = None,
) -> RunProvenance:
    now = now if now is not None else time.time()
    src_env = env if env is not None else dict(os.environ)
    run_id = hashlib.sha256(
        (str(now) + " ".join(flags) + (node_id or "")).encode()
    ).hexdigest()[:12]
    probe_info: Dict[str, Any] = {"present": probe is not None}
    if probe:
        probe_info.update(
            {
                "created": probe.get("created"),
                "driver": probe.get("driver"),
                "uuids": probe.get("uuids"),
                "probe_seconds": probe.get("probe_seconds"),
            }
        )
    budget_info: Dict[str, Any] = {"present": kv_budget is not None}
    if kv_budget is not None:
        budget_info.update(
            {
                "config_hash": getattr(kv_budget, "config_hash", None),
                "max_total_num_tokens": getattr(
                    kv_budget, "max_total_num_tokens", None
                ),
                "age_s": getattr(kv_budget, "age_s", None),
                "warnings": list(getattr(kv_budget, "warnings", [])),
            }
        )
    return RunProvenance(
        run_id=run_id,
        created=now,
        flags=list(flags),
        env={k: src_env[k] for k in SAFE_ENV_KEYS if k in src_env},
        git=git_commit(repo_dir),
        model=model_fingerprint(model_path) if model_path else {},
        draft_model=(
            model_fingerprint(draft_model_path) if draft_model_path else {}
        ),
        probe=probe_info,
        kv_budget=budget_info,
        hardware=[
            {
                "index": c.get("index"),
                "name": c.get("name"),
                "uuid": c.get("uuid"),
                "total_mib": c.get("mem_total_mib"),
            }
            for c in (cards or [])
        ],
        node_id=node_id,
    )


# ---------------------------------------------------------------------------
# State annotation and comparison
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunRecord:
    """A completed run: provenance, headline metrics, and the state it ran in."""

    provenance: RunProvenance
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    state: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "provenance": self.provenance.to_json(),
            "metrics": dict(self.metrics),
            "state": dict(self.state),
        }


def state_summary(cards: Sequence[dict]) -> Dict[str, Any]:
    """Condense the machine state a run happened in.

    ``clean`` is the flag comparisons key off: no performance throttle, and no
    card running far below its maximum clock.
    """
    throttled, low_clock, temps = [], [], []
    for c in cards:
        idx = c.get("index")
        thr = [
            t
            for t in (c.get("throttle") or [])
            if t
            in (
                "sw_thermal_slowdown",
                "hw_thermal_slowdown",
                "sw_power_cap",
                "hw_slowdown",
                "hw_power_brake_slowdown",
            )
        ]
        if thr:
            throttled.append({"gpu": idx, "reasons": thr})
        cur, mx = c.get("sm_clock_mhz"), c.get("sm_clock_max_mhz")
        # 0.90, not a looser bound: the observed thermal case was 1695 of
        # 1905 MHz, i.e. 89 % of maximum. A threshold that let that through
        # would miss the exact situation this check exists for.
        if cur and mx and cur / mx < 0.90:
            low_clock.append({"gpu": idx, "mhz": cur, "max_mhz": mx})
        if c.get("temp_c") is not None:
            temps.append(float(c["temp_c"]))
    return {
        "throttled": throttled,
        "low_clock": low_clock,
        "max_temp_c": max(temps) if temps else None,
        "clean": not throttled and not low_clock,
    }


def compare_runs(
    runs: Sequence[RunRecord],
    metric: str,
    noise_floor_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare a metric across the last N runs, with the state attached.

    A difference is only called a change when the runs were comparable AND the
    difference exceeds the measured noise floor. Without a floor the verdict is
    "unknown", not "changed": a comparison with no threshold reports every
    fluctuation as a finding.
    """
    points = []
    for r in runs:
        if metric not in r.metrics:
            continue
        points.append(
            {
                "run_id": r.provenance.run_id,
                "created": r.provenance.created,
                "value": float(r.metrics[metric]),
                "commit": (r.provenance.git or {}).get("commit"),
                "dirty": (r.provenance.git or {}).get("dirty"),
                "state_clean": bool(r.state.get("clean")),
                "throttled": r.state.get("throttled") or [],
                "max_temp_c": r.state.get("max_temp_c"),
            }
        )
    points.sort(key=lambda p: p["created"])
    out: Dict[str, Any] = {
        "metric": metric,
        "points": points,
        "noise_floor_pct": noise_floor_pct,
    }
    if len(points) < 2:
        out["verdict"] = "insufficient"
        out["explanation"] = "fewer than two runs carry this metric"
        return out

    first, last = points[0], points[-1]
    delta = last["value"] - first["value"]
    pct = (delta / first["value"] * 100.0) if first["value"] else None
    out["delta"] = delta
    out["delta_pct"] = pct

    caveats = []
    if not last["state_clean"] or not first["state_clean"]:
        caveats.append(
            "at least one run was taken under throttling or at a reduced "
            "clock; the difference may be thermal state rather than a change "
            "in the system"
        )
    if last["commit"] != first["commit"]:
        caveats.append("the runs are on different commits")
    if last["dirty"] or first["dirty"]:
        caveats.append(
            "at least one run had a dirty working tree, so its commit does not "
            "identify the code that ran"
        )
    out["caveats"] = caveats

    if noise_floor_pct is None:
        out["verdict"] = "unknown"
        out["explanation"] = (
            "no noise floor recorded, so there is no threshold below which a "
            "difference must not be reported. Run the noise-floor scenario "
            "first."
        )
        return out
    if pct is not None and abs(pct) <= noise_floor_pct:
        out["verdict"] = "within_noise"
        out["explanation"] = (
            f"{pct:+.1f}% is inside the measured noise floor of "
            f"+/-{noise_floor_pct:.1f}%"
        )
        return out
    if caveats:
        out["verdict"] = "not_comparable"
        out["explanation"] = (
            f"{pct:+.1f}% exceeds the noise floor, but the runs are not "
            "comparable: " + "; ".join(caveats)
        )
        return out
    out["verdict"] = "changed"
    out["explanation"] = (
        f"{pct:+.1f}% exceeds the noise floor of +/-{noise_floor_pct:.1f}% and "
        "the runs are comparable"
    )
    return out
