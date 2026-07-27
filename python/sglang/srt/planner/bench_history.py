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
"""On-disk history of benchmark runs, per model.

A pass/fail table is not reviewable on its own. Two months later the question
is never "did test 6 pass" but "what did this model actually answer, and was
that better or worse than the last build". That needs the exchanges kept, not
just the verdicts, and it needs them keyed by model so one model's runs can be
read as a series.

Layout, one JSON file per run::

    <root>/<model-slug>/<timestamp>-<short-id>.json

The model slug is derived from the model reference, so listing a model's
history is a directory read rather than a scan of every run ever made. Each
file holds the whole run: the target, the capabilities it was gated on, every
test result, and the full transcript (request bodies and answers).

Nothing here talks to a server or to the network. The root is
``SGLANG_PLANNER_BENCH_HISTORY`` when set, else a directory under the user's
cache dir.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

__all__ = [
    "history_root",
    "model_slug",
    "save_run",
    "list_runs",
    "load_run",
    "delete_run",
]


def history_root() -> str:
    env = os.environ.get("SGLANG_PLANNER_BENCH_HISTORY")
    if env:
        return os.path.expanduser(env)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "htsglang-planner", "bench_runs")


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def model_slug(model: Optional[str]) -> str:
    """A filesystem-safe directory name for a model reference.

    Only the last two path components are kept, so ``/very/long/path/Model``
    and ``org/Model`` both stay readable in a directory listing. An unnamed
    target lands in ``_unknown`` rather than being dropped -- a run without a
    model name is still a run someone made.
    """
    m = (model or "").strip().rstrip("/")
    if not m:
        return "_unknown"
    parts = [p for p in m.replace("\\", "/").split("/") if p]
    tail = "__".join(parts[-2:]) if len(parts) > 1 else parts[0]
    slug = _SLUG_RE.sub("-", tail).strip("-.")
    return slug or "_unknown"


def _run_dir(model: Optional[str]) -> str:
    return os.path.join(history_root(), model_slug(model))


def save_run(record: Dict[str, Any]) -> str:
    """Persist one finished run and return its id.

    ``record`` is stored as given plus ``run_id`` / ``started_at`` when those
    are missing. Failure to write is raised, not swallowed: a history that
    silently keeps nothing is worse than no history at all.
    """
    model = record.get("model") or record.get("served_model_name")
    started = float(record.get("started_at") or time.time())
    run_id = record.get("run_id") or "{}-{}".format(
        time.strftime("%Y%m%dT%H%M%S", time.localtime(started)),
        uuid.uuid4().hex[:8],
    )
    out = dict(record)
    out["run_id"] = run_id
    out["started_at"] = started
    out["model_slug"] = model_slug(model)
    d = _run_dir(model)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, run_id + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, path)
    return run_id


def _summary(rec: Dict[str, Any], path: str) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rec.get("results") or []:
        st = r.get("status") or "unknown"
        counts[st] = counts.get(st, 0) + 1
    return {
        "run_id": rec.get("run_id"),
        "model": rec.get("model"),
        "model_slug": rec.get("model_slug"),
        "endpoint": rec.get("endpoint"),
        "started_at": rec.get("started_at"),
        "duration_s": rec.get("duration_s"),
        "preset": rec.get("preset"),
        "counts": counts,
        "n_tests": len(rec.get("results") or []),
        "n_exchanges": len(rec.get("transcript") or []),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
    }


def list_runs(model: Optional[str] = None, limit: int = 50) -> List[dict]:
    """Newest-first summaries. With no ``model`` every model's runs are
    merged; a corrupt file is skipped, never fatal."""
    roots = []
    if model:
        roots.append(_run_dir(model))
    else:
        base = history_root()
        try:
            roots = [os.path.join(base, d) for d in sorted(os.listdir(base))]
        except OSError:
            roots = []
    out: List[dict] = []
    for d in roots:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d), reverse=True):
            if not name.endswith(".json"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            out.append(_summary(rec, path))
    out.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return out[:limit]


def _find_path(run_id: str) -> Optional[str]:
    base = history_root()
    try:
        subs = os.listdir(base)
    except OSError:
        return None
    for sub in subs:
        p = os.path.join(base, sub, run_id + ".json")
        if os.path.exists(p):
            return p
    return None


def load_run(run_id: str) -> Optional[dict]:
    """The complete stored run, transcript included, or None."""
    # A run id is a file name component; anything with a separator in it is a
    # path traversal attempt, not a run id.
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    path = _find_path(run_id)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_run(run_id: str) -> bool:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return False
    path = _find_path(run_id)
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False
