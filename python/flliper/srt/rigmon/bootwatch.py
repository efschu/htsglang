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
"""Boot watching: follow a start-up and classify how it failed.

Most of the pain is at boot, and a spinner that turns into "failed" is the
least useful possible rendering of it. This module reads the boot log as it
grows, tracks how far start-up got, and matches known failure signatures
against the tail.

The classification is worth more than the raw traceback because the same
Python exception means different things at different stages. An
``OutOfMemoryError`` while loading weights says the model does not fit; the
same exception while allocating the KV pool says the memory budget was
mis-planned and the weights were fine. Stage plus signature, not signature
alone.

Patterns below were taken from real boot logs on this rig, not invented.

Reading is **streamed and bounded**. Crash-looping servers have produced logs
of many gigabytes; a classifier that slurps the file is a second outage.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "BootStage",
    "BootDiagnosis",
    "Signature",
    "SIGNATURES",
    "STAGES",
    "classify_boot",
    "read_boot_log",
]


# ---------------------------------------------------------------------------
# Progress stages
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BootStage:
    key: str
    label: str
    marker: str
    order: int


#: Ordered start-up milestones, keyed off lines the server actually prints.
STAGES: Tuple[BootStage, ...] = (
    BootStage("launch", "process launched", "Launch server", 0),
    BootStage("weights_begin", "loading weights", "Load weight begin", 1),
    BootStage("weights_done", "weights loaded", "Load weight end", 2),
    BootStage("kv_allocated", "KV cache allocated", "KV Cache is allocated", 3),
    BootStage("graphs", "capturing CUDA graphs", "Capture cuda graph", 4),
    BootStage("ready", "serving", "fired up and ready", 5),
)


# ---------------------------------------------------------------------------
# Failure signatures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Signature:
    """A known failure, with what it means and what to change.

    ``after_stage`` narrows a generic exception to a specific meaning: OOM
    before the weights are loaded is a different problem from OOM after.
    ``priority`` breaks ties — a specific signature must win over the generic
    "a child process died" that accompanies almost every failure.
    """

    key: str
    label: str
    pattern: str
    meaning: str
    remedy: str
    priority: int = 50
    after_stage: Optional[str] = None
    before_stage: Optional[str] = None
    severity: str = "error"

    def compiled(self):
        return re.compile(self.pattern)


SIGNATURES: Tuple[Signature, ...] = (
    Signature(
        key="oom_pools",
        label="out of memory while sizing the caches",
        pattern=r"(torch\.)?OutOfMemoryError|CUDA out of memory",
        after_stage="weights_done",
        priority=90,
        meaning=(
            "The weights loaded, then the KV / mamba pool allocation ran out "
            "of room. The model fits; the memory budget for this rank does "
            "not. This is a planning failure, not a capacity one."
        ),
        remedy=(
            "Lower the per-rank memory budget or the context target, or "
            "re-plan the split. If the persisted measured KV budget is in "
            "play, check it: a budget recorded under a different boot order "
            "can shift max_total_num_tokens substantially."
        ),
    ),
    Signature(
        key="oom_weights",
        label="out of memory while loading weights",
        pattern=r"(torch\.)?OutOfMemoryError|CUDA out of memory",
        before_stage="weights_done",
        priority=85,
        meaning=(
            "Ran out of VRAM before the weights were in. The shard assigned "
            "to this rank does not fit on its card at all."
        ),
        remedy=(
            "Re-plan the split, use a smaller quantisation, or place fewer "
            "ranks on the card. Raising the context target will not help."
        ),
    ),
    Signature(
        key="arch_mismatch",
        label="kernel built for a different architecture",
        pattern=r"cudaErrorNoKernelImageForDevice|no kernel image is available",
        priority=95,
        meaning=(
            "A compiled kernel carries no cubin for this card's architecture. "
            "On a mixed rig this typically means an extension was built for "
            "one card and reused on the other."
        ),
        remedy=(
            "Rebuild the extension with every architecture on this rig in the "
            "arch list, or clear the JIT cache and let it rebuild. Check that "
            "the cache key includes the architecture — a key without it is the "
            "recurring form of this failure."
        ),
    ),
    Signature(
        key="poisoned_cache",
        label="stale or mismatched compiled extension",
        pattern=r"undefined symbol|ImportError: .*\.so|version `GLIBCXX",
        priority=93,
        meaning=(
            "A compiled extension does not match the runtime that loads it — "
            "typically a cached build from a different torch or toolkit "
            "version."
        ),
        remedy=(
            "Clear the extension / JIT cache and rebuild against the current "
            "environment."
        ),
    ),
    Signature(
        key="mamba_pool_too_small",
        label="hybrid state cache too small to serve",
        # Anchored on the error sentence only. An earlier draft also matched
        # `max_mamba_`, which occurs in ordinary configuration lines and fired
        # on 96 of 116 real logs, including successful boots.
        pattern=r"state cache is too small to serve any requests",
        priority=88,
        meaning=(
            "The recurrent-state pool came out below one request's worth. The "
            "memory budget was consumed by weights and the KV pool before the "
            "state pool got its share."
        ),
        remedy=(
            "Raise the rank memory budget, lower the concurrent-request "
            "target, or shift weight mass off this rank."
        ),
    ),
    Signature(
        key="geometry_mismatch",
        label="model geometry not divisible by the split",
        pattern=r"(\d+) is not divisible by (\d+)|not divisible by",
        priority=87,
        meaning=(
            "A head, expert or vocabulary count does not divide by the "
            "requested parallel width. Even splits require divisibility; the "
            "uneven-TP path exists precisely for the counts that do not "
            "divide."
        ),
        remedy=(
            "Choose a parallel width that divides the count, or use the "
            "uneven-TP split with an explicit per-rank ratio."
        ),
    ),
    Signature(
        key="backend_unsupported",
        label="attention backend not supported by this model",
        pattern=r"only supports .* attention backend|AssertionError: .*attention backend",
        priority=86,
        meaning="The chosen attention backend does not implement this model.",
        remedy="Select one of the backends named in the message.",
    ),
    Signature(
        key="nccl_error",
        label="collective initialisation failed",
        pattern=r"NCCL error|ncclInternalError|ncclInvalidUsage",
        priority=80,
        meaning=(
            "The process group failed to form or a collective was mis-issued. "
            "On this rig the frequent cause is a rank-local condition that "
            "aborts one rank before a collective, leaving the others waiting."
        ),
        remedy=(
            "Run with NCCL_DEBUG=WARN and look for the FIRST rank to fail: the "
            "NCCL error is usually the symptom of another rank's earlier, "
            "rank-local failure."
        ),
    ),
    Signature(
        key="collective_hang",
        label="collective timed out",
        pattern=r"Watchdog caught collective operation timeout|NCCL.*[Tt]imeout",
        priority=82,
        meaning=(
            "One rank never reached a collective the others entered. Ranks "
            "diverged in control flow, or one died silently."
        ),
        remedy=(
            "Check every rank's log for the earliest error. A rank-local "
            "condition evaluated before a group collective is the recurring "
            "shape of this failure."
        ),
    ),
    Signature(
        key="port_in_use",
        label="port already bound",
        pattern=r"Address already in use|EADDRINUSE",
        priority=91,
        meaning="Another server (often a previous run) still holds the port.",
        remedy=(
            "Stop the previous server, or pick another port. Check for orphan "
            "processes still holding VRAM."
        ),
    ),
    Signature(
        key="model_not_found",
        label="model path unreadable",
        pattern=r"No such file or directory: .*(config\.json|\.gguf|\.safetensors)"
        r"|does not appear to have a file named config\.json",
        priority=92,
        meaning="The model path is wrong or not mounted in this container.",
        remedy="Correct the path, or mount the model directory.",
    ),
    Signature(
        key="child_died",
        label="a worker process died during initialisation",
        pattern=r"scheduler died during initialization|Received sigquit from a child",
        priority=10,
        meaning=(
            "A worker exited during start-up. This line accompanies almost "
            "every boot failure and is the symptom, not the cause."
        ),
        remedy=(
            "Look further up the log for the first error. If the exit code is "
            "-9 the kernel OOM-killer took the process: that is HOST memory, "
            "not VRAM."
        ),
        severity="warning",
    ),
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

#: Never read more than this from a log; crash loops have produced multi-GB
#: files and a classifier that slurps one is a second outage.
MAX_READ_BYTES = 32 * 1024 * 1024
#: Failure signatures are matched against the tail; progress markers against
#: the whole (bounded) read.
TAIL_BYTES = 512 * 1024


def read_boot_log(
    path: str, max_bytes: int = MAX_READ_BYTES, tail_bytes: int = TAIL_BYTES
) -> Tuple[List[str], List[str]]:
    """Return ``(marker_lines, tail_lines)``.

    Progress markers are short and appear early, so the head is scanned for
    them; failures appear last, so the tail is kept in full.
    """
    size = os.path.getsize(path)
    markers: List[str] = []
    marker_texts = [s.marker for s in STAGES]
    with open(path, "rb") as f:
        head = f.read(min(size, max_bytes))
        for raw in head.decode("utf-8", "replace").splitlines():
            if any(m in raw for m in marker_texts):
                markers.append(raw)
        if size > tail_bytes:
            f.seek(max(0, size - tail_bytes))
            tail_blob = f.read()
        else:
            tail_blob = head
    tail = tail_blob.decode("utf-8", "replace").splitlines()
    # The tail can also carry markers when the log is longer than max_bytes.
    for raw in tail:
        if any(m in raw for m in marker_texts) and raw not in markers:
            markers.append(raw)
    return markers, tail


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BootDiagnosis:
    """How far the boot got, and what stopped it."""

    stage: str
    stage_label: str
    stage_order: int
    reached: List[str]
    ready: bool
    failed: bool
    diagnosis: Optional[Dict[str, str]] = None
    other_matches: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    excerpt: List[str] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _stage_order(key: Optional[str]) -> int:
    for s in STAGES:
        if s.key == key:
            return s.order
    return -1


def classify_boot(
    marker_lines: List[str], tail_lines: List[str]
) -> BootDiagnosis:
    """Stage plus signature. Both are needed: the same exception means
    different things depending on how far start-up got."""
    reached = []
    current = STAGES[0]
    for s in STAGES:
        if any(s.marker in line for line in marker_lines):
            reached.append(s.key)
            if s.order >= current.order:
                current = s
    ready = "ready" in reached

    blob = "\n".join(tail_lines)
    matches = []
    for sig in SIGNATURES:
        if not sig.compiled().search(blob):
            continue
        if sig.after_stage and current.order < _stage_order(sig.after_stage):
            continue
        if sig.before_stage and current.order >= _stage_order(sig.before_stage):
            continue
        matches.append(sig)
    matches.sort(key=lambda s: -s.priority)

    failed = bool(matches) and not ready
    primary = None
    if ready:
        # The boot succeeded. Signatures still matching in the tail belong to
        # whatever happened afterwards, and reporting one as the boot's
        # diagnosis would turn a healthy start into a phantom failure.
        matches = []
    if matches:
        s = matches[0]
        primary = {
            "key": s.key,
            "label": s.label,
            "meaning": s.meaning,
            "remedy": s.remedy,
            "severity": s.severity,
            "stage": current.key,
        }
    excerpt = [
        ln
        for ln in tail_lines[-40:]
        if ln.strip()
        and any(t in ln for t in ("Error", "error", "Traceback", "Assertion", "kill"))
    ][-12:]
    return BootDiagnosis(
        stage=current.key,
        stage_label=current.label,
        stage_order=current.order,
        reached=reached,
        ready=ready,
        failed=failed,
        diagnosis=primary,
        other_matches=[
            {"key": s.key, "label": s.label} for s in matches[1:]
        ],
        excerpt=excerpt,
    )
