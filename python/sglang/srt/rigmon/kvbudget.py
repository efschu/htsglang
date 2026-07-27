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
"""The persisted measured KV budget: visible, explained, resettable.

``~/.cache/sglang/kv_budget-<confighash>.json`` records what a previous boot
measured as actually available. The next boot with a matching fingerprint
reuses it — which is the point, and also the trap: the file records the state
of the machine at the moment it was written. A budget measured while another
process still held VRAM carries that occupancy forward into every later boot,
and ``max_total_num_tokens`` moves by a large factor for reasons invisible in
the command line. A shift of roughly 4x has been observed from boot order
alone.

"Why is my context different today" is a question a user will ask, and the
answer lives in this file. So it gets three things: it is listed, it is
explained in terms of the numbers that produced the budget, and it can be
cleared — with a backup, because clearing it forces a re-measurement that is
only as good as the state of the machine at that moment.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

__all__ = [
    "BudgetFile",
    "CACHE_DIR",
    "list_budget_files",
    "describe_budget",
    "reset_budget",
]

CACHE_DIR = os.path.expanduser("~/.cache/sglang")

_MIB = 1024.0 * 1024.0


@dataclasses.dataclass
class BudgetFile:
    path: str
    config_hash: str
    mtime: float
    age_s: float
    size_bytes: int
    max_total_num_tokens: Optional[int] = None
    ranks: int = 0
    safety_mib: Optional[int] = None
    #: Per-rank breakdown of what the measurement attributed the memory to.
    components: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    raw: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("raw", None)
        return d


def list_budget_files(cache_dir: str = CACHE_DIR) -> List[str]:
    try:
        names = sorted(os.listdir(cache_dir))
    except OSError:
        return []
    return [
        os.path.join(cache_dir, n)
        for n in names
        if n.startswith("kv_budget-") and n.endswith(".json")
    ]


def _f(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def describe_budget(path: str, now: Optional[float] = None) -> BudgetFile:
    """Read one budget file and explain what it fixed.

    The per-rank breakdown is the part that answers the user's question: it
    shows how much free memory the measuring boot SAW, which is the number a
    competing process would have distorted.
    """
    now = now if now is not None else time.time()
    st = os.stat(path)
    name = os.path.basename(path)
    config_hash = name[len("kv_budget-") : -len(".json")]
    with open(path) as f:
        data = json.load(f)

    out = BudgetFile(
        path=path,
        config_hash=config_hash,
        mtime=st.st_mtime,
        age_s=max(0.0, now - st.st_mtime),
        size_bytes=st.st_size,
        safety_mib=data.get("safety_mib"),
        raw=data,
    )

    comps = data.get("components") or []
    out.ranks = len(comps)
    tokens = {c.get("max_total_num_tokens") for c in comps if c.get("max_total_num_tokens")}
    if len(tokens) == 1:
        out.max_total_num_tokens = int(next(iter(tokens)))
    elif tokens:
        out.max_total_num_tokens = int(min(tokens))
        out.warnings.append(
            "ranks disagree on max_total_num_tokens "
            f"({sorted(int(t) for t in tokens)}); the minimum is what the pool "
            "is synced to"
        )

    for i, c in enumerate(comps):
        free = _f(c, "free_bytes_at_measure")
        total = _f(c, "device_total_bytes")
        resid = _f(c, "residual_residency_bytes")
        entry = {
            "rank": i,
            "device_total_mib": round(total / _MIB) if total else None,
            "free_at_measure_mib": round(free / _MIB) if free is not None else None,
            "weights_mib": (
                round(_f(c, "weights_alloc_bytes") / _MIB)
                if _f(c, "weights_alloc_bytes")
                else None
            ),
            "kv_pool_mib": (
                round(_f(c, "kv_pool_bytes") / _MIB) if _f(c, "kv_pool_bytes") else None
            ),
            "kv_pool_tokens": c.get("kv_pool_tokens"),
            "mamba_pool_mib": (
                round(_f(c, "mamba_aux_pool_bytes") / _MIB)
                if _f(c, "mamba_aux_pool_bytes")
                else None
            ),
            "graphs_mib": (
                round(_f(c, "graphs_ws_bytes") / _MIB)
                if _f(c, "graphs_ws_bytes")
                else None
            ),
            "fragmentation_mib": (
                round(_f(c, "frag_bytes") / _MIB) if _f(c, "frag_bytes") else None
            ),
            # The load-bearing field: memory that was resident and NOT this
            # process's. If it is large, a competing process was running when
            # the budget was measured, and every later boot inherits that.
            "foreign_residency_mib": round(resid / _MIB) if resid is not None else None,
            "ranks_on_gpu": c.get("ranks_on_gpu"),
            "safety_mib": c.get("safety_mib"),
        }
        out.components.append(entry)
        if resid is not None and resid > 1.5 * 2**30:
            out.warnings.append(
                f"rank {i}: {resid / 2**30:.1f} GiB was resident on the card and "
                "not owned by the measuring process. The budget recorded here "
                "carries that occupancy forward into every later boot with this "
                "fingerprint; if that process is gone, reset the file and let "
                "it re-measure."
            )

    if out.age_s > 30 * 86400:
        out.warnings.append(
            f"budget is {out.age_s / 86400:.0f} days old; the hardware or the "
            "driver may have changed since"
        )
    return out


def reset_budget(path: str, backup: bool = True) -> Dict[str, Any]:
    """Remove a budget file so the next boot re-measures.

    A backup is kept by default: the re-measurement is itself only as good as
    the machine's state at that moment, so throwing away a known-good record
    without a copy trades one unexplained number for another.
    """
    if not os.path.isfile(path):
        return {"removed": False, "reason": f"no such file: {path}"}
    backup_path = None
    if backup:
        backup_path = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup_path)
    os.remove(path)
    return {
        "removed": True,
        "path": path,
        "backup": backup_path,
        "effect": (
            "The next boot with this fingerprint re-measures the budget. Make "
            "sure no other process holds VRAM at that moment, or the new value "
            "will carry that occupancy instead."
        ),
    }
