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
"""The VRAM flight recorder: measure what a rank's boot actually costs, per post.

WHY THIS EXISTS. The ledger's internal demand is MODELED, and #602 measured how
far the model is from the truth: 4664 / 993 / 2701 MiB of overprediction on the
three cards of the reference rig, i.e. up to 4.7 GiB of KV pool given away per
card. A modeled term can only be corrected against a MEASURED one, and no
instrument in the tree produced per-post resident numbers. This module is that
instrument. It measures, it logs, and it attributes; it does not size anything.

THREE SOURCES, BECAUSE NO SINGLE ONE COVERS THE POSTS
-----------------------------------------------------

**1. Phase marks (primary).** :func:`mark` reads ``torch.cuda.memory_stats``
and NVML at named boot boundaries and appends one JSON line per mark. The
difference between two consecutive marks IS the post that was allocated between
them -- weights, KV pool, capture pool, workspaces -- measured rather than
inferred, at a cost of one NVML call. Cheap enough to leave on.

**2. Process-start allocation recording (corroboration + call sites).**
:func:`arm_process_trace` turns on ``torch.cuda.memory._record_memory_history``
BEFORE the first CUDA allocation of the process. This is the only way to learn
WHICH LINE allocated a resident post. Two properties of torch's snapshot format
decide the design, and both were verified against the #602 captures:

* ``segments[].blocks[].frames`` is populated **only for blocks allocated after
  recording began**. In the #602 captures -- recording armed post-boot via
  ``/start_profile`` -- that was 80 of 2046 blocks, 3 MiB of 25142 MiB reserved.
  The structure was not the wrong one to read; it was STARVED. Armed at process
  start it becomes the exact resident attribution, because it is keyed on the
  blocks the allocator is actually still holding.
* ``device_traces`` is a fixed-size RING. ``max_entries`` is therefore never
  capped here (torch's default is effectively unbounded): the #602 capture came
  back with exactly 100000 entries, i.e. full, i.e. wrapped, retaining only the
  last ~10.7 s of a boot-long history.

Neither structure is trusted blind: :func:`resident_attribution` and
:func:`churn_attribution` each return an explicit COVERAGE verdict and refuse to
present a partial answer as a total. See :class:`Coverage`.

**3. The non-torch remainder, from outside torch.** torch cannot see its own
CUDA context, the driver's BAR1 windows, or a workspace a backend allocated with
raw ``cudaMalloc``. Every mark therefore also records what NVML says THIS PID
holds on the card, so

    non_torch_bytes = nvml_bytes_for_this_pid - torch_reserved_bytes

is a measured quantity per phase rather than a gap between two card-level
totals. The per-PID reading also settles the parent/tokenizer context question
(``TERM_PARENT_CONTEXT``) by DIRECT observation -- the parent either appears in
NVML's per-process list on this card or it does not -- instead of inferring it
from one card sitting higher than its siblings.

WHAT THIS MODULE MAY NOT DO. It records. It never sizes a pool, never adjusts a
term, and never writes a number into the ledger. The path from these
measurements to demand model v2 runs through the fingerprinted store, where a
measurement is valid only for the hardware x model x quant x config it was taken
under; a number from here that is carried to another rig is a guess wearing a
measurement's clothes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "TRACE_ENV",
    "DIR_ENV",
    "Coverage",
    "SiteFootprint",
    "arm_process_trace",
    "trace_requested_for_rank",
    "disarm_process_trace",
    "trace_armed",
    "dump_trace",
    "is_recording_phases",
    "mark",
    "read_marks",
    "phase_deltas",
    "python_site",
    "python_stack",
    "resident_attribution",
    "churn_attribution",
]

MIB = 1 << 20

#: Arms source 2 (process-start allocation recording). Set on the SERVER
#: process; every scheduler process inherits it. Off by default because the
#: recording costs host RAM proportional to the number of allocations a boot
#: makes, and that cost is a measurement's price, not a serving cost.
#: ``1``/``all`` arms every rank; a comma-separated rank list arms only those,
#: which is how a first measurement boot keeps its host-RAM cost to one rank.
TRACE_ENV = "SGLANG_VRAM_FLIGHT_TRACE"

#: Directory for the phase-mark log (source 1) and any snapshot dumps. Absent,
#: every entry point in this module returns immediately and allocates nothing.
DIR_ENV = "SGLANG_VRAM_FLIGHT_DIR"

#: The boot boundaries, in order, and what the gap AFTER each one contains.
#: This is the contract between the instrument and its reader, and a
#: registered test asserts that the serving tree calls :func:`mark` with
#: exactly these names -- no more, so a typo cannot invent a phase, and no
#: fewer, so a deleted call site cannot quietly shrink the record. #602's
#: lesson: twelve green fixture tests passed while the production carrier
#: lacked the field they all built by hand.
BOOT_PHASES: Tuple[Tuple[str, str], ...] = (
    ("process_start", "before any CUDA allocation; the context is not yet bound"),
    ("pre_weight_load", "CUDA context, comm init, allocator warm-up"),
    ("weights_loaded", "the model shard this rank holds"),
    ("kv_pool_sized", "the KV pool and the mamba/GDN state pool"),
    ("capture_begin", "attention backends and their workspaces"),
    ("capture_end", "the CUDA graph pool (once per runner: target, draft)"),
    ("boot_complete", "everything the remaining boot steps allocate"),
    ("first_forward", "the prefill transient, on top of the resident set"),
)


# ---------------------------------------------------------------------------
# Source 2: process-start allocation recording
# ---------------------------------------------------------------------------

_trace_armed = False


def trace_armed() -> bool:
    return _trace_armed


def trace_requested_for_rank(rank: int) -> bool:
    """Whether :data:`TRACE_ENV` asks for a trace on THIS rank.

    ``1``/``all`` arms every rank; a comma-separated rank list arms only those.
    A SCOPE, not a cap: whichever ranks are armed record their whole boot with
    nothing dropped. It exists because the ring is uncapped by design and its
    host-RAM cost scales with a boot's allocation count on a swapless box, so
    "measure one rank first" has to be expressible without reaching for
    ``max_entries`` -- which is the knob that produced the #602 wrap.
    """
    raw = (os.environ.get(TRACE_ENV) or "").strip()
    if not raw:
        return False
    if raw.lower() in {"1", "all", "true", "yes"}:
        return True
    wanted = {tok.strip() for tok in raw.split(",") if tok.strip()}
    return str(int(rank)) in wanted


def arm_process_trace(rank: int = 0, force: bool = False) -> bool:
    """Start recording allocations, before the process's first CUDA allocation.

    ``max_entries`` is deliberately NOT passed. The #602 capture set it to
    100000, which is a ring of 100000 events, and a boot makes more than that:
    the captured trace came back exactly full, holding only its final 10.7
    seconds. A ring that wraps silently turns "this post has no allocation
    event" into an untrue statement, and this instrument's whole value is that
    its absences are honest.

    Returns True when recording was turned on. Safe to call more than once.
    """
    global _trace_armed
    if _trace_armed:
        return True
    if not (force or trace_requested_for_rank(rank)):
        return False
    try:
        import torch

        # enabled/context/stacks are left at their defaults on purpose: they
        # already are 'all'/'all'/'all'. Naming them would suggest that the
        # #602 capture lacked stacks because of a flag, which it did not.
        torch.cuda.memory._record_memory_history()
    except Exception as e:  # pragma: no cover - torch/platform differences
        logger.warning("VRAM flight recorder could not arm the trace: %s", e)
        return False
    _trace_armed = True
    logger.info(
        "VRAM flight recorder: allocation recording armed for this process "
        "(uncapped ring). Host RAM grows with the number of allocations; this "
        "is a measurement boot, not a serving configuration."
    )
    return True


def disarm_process_trace() -> bool:
    """Stop recording. Called once the boot snapshot has been taken.

    NOT a cap in disguise. ``max_entries`` stays uncapped, so nothing inside
    the recorded window is ever silently dropped -- the failure mode of the
    #602 capture. What ends here is the WINDOW: the posts this instrument
    exists to attribute are all resident by graph-capture end, while serving
    goes on allocating at 3.6 us instead of 2.1 us per allocation (measured on
    the reference rig) into a ring that grows for the life of the process. A
    boot-length window bounded by an event is the honest way to bound it; a
    fixed entry count is not.
    """
    global _trace_armed
    if not _trace_armed:
        return False
    try:
        import torch

        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as e:  # pragma: no cover - torch/platform differences
        logger.warning("VRAM flight recorder could not disarm the trace: %s", e)
        return False
    _trace_armed = False
    logger.info(
        "VRAM flight recorder: allocation recording stopped; the boot window "
        "is closed and serving runs at full speed again."
    )
    return True


def dump_trace(
    tag: str, *, rank: int = 0, directory: Optional[str] = None
) -> Optional[str]:
    """Write the allocation snapshot. No-op unless the trace was armed."""
    if not _trace_armed:
        return None
    directory = directory or os.environ.get(DIR_ENV)
    if not directory:
        return None
    try:
        import torch

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"flight_trace_rank{rank}_{tag}.pickle")
        torch.cuda.memory._dump_snapshot(path)
        logger.info("VRAM flight recorder: snapshot %s written to %s", tag, path)
        return path
    except Exception as e:  # pragma: no cover - torch/filesystem differences
        logger.warning("VRAM flight recorder could not dump snapshot %s: %s", tag, e)
        return None


# ---------------------------------------------------------------------------
# Source 1 + 3: phase marks
# ---------------------------------------------------------------------------


def is_recording_phases() -> bool:
    return bool(os.environ.get(DIR_ENV))


def cuda_initialized() -> bool:
    """True when torch's lazy CUDA init has run in this process.

    THIS IS NOT "a primary context exists", and the difference is load-bearing
    enough to be measured rather than assumed. On the reference rig (5090,
    driver 5xx, torch 2.x), with ``CUDA_VISIBLE_DEVICES`` pinned to one card:

        after ``import torch``                      is_initialized False, NVML   0 MiB
        after importing any ``sglang.srt`` module   is_initialized True,  NVML   0 MiB
        after the first 1-byte allocation           is_initialized True,  NVML 500 MiB

    So an sglang import flips the flag while binding nothing, and the 500 MiB
    context is bought by the first ALLOCATION. Verified in the same run that
    none of ``current_device()``, ``memory_stats()`` or
    ``_record_memory_history()`` binds it either -- which is what makes it safe
    for a mark to read them at process start.

    The direct question ("does this process hold memory on that card") is
    answered by NVML's per-process list, and every mark records it as
    :data:`holds_device_context`. Inferring it from this flag would be exactly
    the gap-inference this instrument exists to replace.
    """
    try:
        import torch

        return bool(torch.cuda.is_initialized())
    except Exception:  # pragma: no cover - torch-less hosts
        return False


def _torch_view(device_index: Optional[int]) -> Dict[str, Any]:
    if not cuda_initialized():
        return {"cuda_initialized": False}
    try:
        import torch

        if device_index is None:
            device_index = int(torch.cuda.current_device())
        stats = torch.cuda.memory_stats(device_index)
    except Exception as e:  # pragma: no cover - non-CUDA hosts
        logger.debug("flight recorder: no torch memory stats (%s)", e)
        return {"cuda_initialized": True}
    return {
        "cuda_initialized": True,
        "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
        "allocated_peak_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
        "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
        "reserved_peak_bytes": int(stats.get("reserved_bytes.all.peak", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "num_ooms": int(stats.get("num_ooms", 0)),
    }


def _nvml_view() -> Dict[str, Any]:
    """This rank's card as the DRIVER sees it, plus who else is on it.

    Cards are identified by UUID, never by index: NVML and torch order devices
    differently and a masked worker sees a different index again.
    """
    try:
        from sglang.srt.registry import nvml as registry_nvml

        # Resolving WHICH card this process is pinned to falls back to torch
        # when the pin cannot be read from the environment, and that fallback
        # initialises CUDA. Before the context exists, decline instead: an
        # unnamed card is an honest absence, a context created by the
        # instrument is a corrupted measurement.
        if not cuda_initialized() and not registry_nvml.pin_resolvable_without_cuda():
            return {
                "nvml_card_unresolved": "no CUDA context yet and the pin is "
                "not readable from the environment; refusing to create a "
                "context in order to name the card"
            }
        uuid = registry_nvml.current_device_uuid()
        info = registry_nvml.memory_info_for_uuid(uuid)
        procs = registry_nvml.process_bytes_on_uuid(uuid)
    except Exception as e:
        logger.debug("flight recorder: no NVML view (%s)", e)
        return {"nvml_card_unresolved": str(e)}
    pid = os.getpid()
    return {
        "card_uuid": uuid,
        "nvml_total_bytes": int(info.total_bytes),
        "nvml_free_bytes": int(info.free_bytes),
        "nvml_used_bytes": int(info.used_bytes),
        "nvml_carve_out_bytes": int(info.reserved_bytes),
        # The per-PID reading is what makes the non-torch remainder a
        # measurement of THIS process rather than a card-level leftover that a
        # co-located rank or the parent would pollute.
        "nvml_self_bytes": int(procs.get(pid, 0)),
        "nvml_processes": {str(k): int(v) for k, v in sorted(procs.items())},
    }


def mark(
    phase: str,
    *,
    rank: int = 0,
    device_index: Optional[int] = None,
    directory: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[dict]:
    """Record one phase boundary. No-op unless :data:`DIR_ENV` is set.

    Appended, never rewritten: a rank that dies at graph capture must still
    leave behind every boundary it did reach. That is the difference between a
    flight recorder and a report.
    """
    directory = directory or os.environ.get(DIR_ENV)
    if not directory:
        return None
    record: Dict[str, Any] = {
        "phase": str(phase),
        "rank": int(rank),
        "pid": os.getpid(),
        "wall": time.time(),
        "monotonic": time.monotonic(),
    }
    record.update(_torch_view(device_index))
    record.update(_nvml_view())
    if "nvml_self_bytes" in record:
        # The DIRECT check for "has this process bound a primary context yet",
        # as opposed to torch's is_initialized flag, which flips on an import
        # and says nothing about the card. See :func:`cuda_initialized`.
        record["holds_device_context"] = int(record["nvml_self_bytes"]) > 0
    if "nvml_self_bytes" in record and "reserved_bytes" in record:
        # The whole point of source 3, computed at write time so a reader of
        # one line does not have to know the subtraction. Floored at 0: NVML
        # rounds to MiB granularity and torch does not, so a sub-MiB negative
        # here is quantisation, not a discovery.
        record["non_torch_bytes"] = max(
            0, int(record["nvml_self_bytes"]) - int(record["reserved_bytes"])
        )
    if extra:
        record["extra"] = dict(extra)
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"flight_marks_rank{rank}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:  # pragma: no cover - filesystem differences
        logger.warning("VRAM flight recorder could not write a phase mark: %s", e)
        return record
    return record


def read_marks(directory: str) -> Dict[int, List[dict]]:
    """``{rank: [mark, ...]}`` in the order they were written."""
    out: Dict[int, List[dict]] = {}
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("flight_marks_rank") and name.endswith(".jsonl")):
            continue
        path = os.path.join(directory, name)
        records: List[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    # A torn final line is what a crash mid-write looks like.
                    # Keep the boundaries that did land rather than dropping
                    # the rank's whole history over its last byte.
                    logger.warning("flight recorder: dropping a torn line in %s", path)
        if records:
            out[int(records[0].get("rank", 0))] = records
    return out


@dataclasses.dataclass(frozen=True)
class PhaseDelta:
    """What was allocated BETWEEN two consecutive marks."""

    frm: str
    to: str
    torch_reserved_bytes: int
    torch_allocated_bytes: int
    non_torch_bytes: int
    nvml_self_bytes: int
    seconds: float

    @property
    def total_bytes(self) -> int:
        return self.torch_reserved_bytes + self.non_torch_bytes

    def row(self) -> str:
        return (
            f"{self.frm} -> {self.to}: torch reserved "
            f"{self.torch_reserved_bytes // MIB:+d} MiB, non-torch "
            f"{self.non_torch_bytes // MIB:+d} MiB, NVML self "
            f"{self.nvml_self_bytes // MIB:+d} MiB, {self.seconds:.1f} s"
        )


def phase_deltas(marks: Sequence[Mapping[str, Any]]) -> List[PhaseDelta]:
    """Consecutive differences. Each one is a post, measured."""
    out: List[PhaseDelta] = []
    for a, b in zip(marks, marks[1:]):
        out.append(
            PhaseDelta(
                frm=str(a.get("phase", "?")),
                to=str(b.get("phase", "?")),
                torch_reserved_bytes=int(b.get("reserved_bytes", 0))
                - int(a.get("reserved_bytes", 0)),
                torch_allocated_bytes=int(b.get("allocated_bytes", 0))
                - int(a.get("allocated_bytes", 0)),
                non_torch_bytes=int(b.get("non_torch_bytes", 0))
                - int(a.get("non_torch_bytes", 0)),
                nvml_self_bytes=int(b.get("nvml_self_bytes", 0))
                - int(a.get("nvml_self_bytes", 0)),
                seconds=float(b.get("monotonic", 0.0)) - float(a.get("monotonic", 0.0)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Snapshot attribution
# ---------------------------------------------------------------------------


def python_site(frames: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """``file.py:line function`` of the ALLOCATING python frame, or None.

    torch's unwind interleaves C++ and CPython frames innermost-first, so the
    allocating python line is the first entry whose filename is a ``.py`` --
    not the first entry, and not the last python one, which is the process
    entry point.
    """
    for frame in frames or ():
        filename = str(frame.get("filename") or "")
        if filename.endswith(".py"):
            return f"{filename}:{frame.get('line', 0)} {frame.get('name', '')}".strip()
    return None


def python_stack(
    frames: Iterable[Mapping[str, Any]], limit: int = 8
) -> Tuple[str, ...]:
    """The python frames only, innermost first, for reading a site in context."""
    out: List[str] = []
    for frame in frames or ():
        filename = str(frame.get("filename") or "")
        if not filename.endswith(".py"):
            continue
        out.append(f"{filename}:{frame.get('line', 0)} {frame.get('name', '')}".strip())
        if len(out) >= limit:
            break
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class Coverage:
    """How much of the quantity the attribution actually explains.

    Exists because the #602 capture produced a 3 MiB answer for a 25142 MiB
    question and the answer looked like a total. An attribution that cannot say
    what it missed is not a measurement.
    """

    attributed_bytes: int
    total_bytes: int
    attributed_items: int
    total_items: int
    #: Set when the evidence says the record does not reach process start.
    starts_after_process_start: bool = False
    note: str = ""

    @property
    def complete(self) -> bool:
        return self.attributed_bytes >= self.total_bytes and not (
            self.starts_after_process_start
        )

    @property
    def missing_bytes(self) -> int:
        return max(0, self.total_bytes - self.attributed_bytes)

    @property
    def fraction(self) -> float:
        return (self.attributed_bytes / self.total_bytes) if self.total_bytes else 1.0

    def verdict(self) -> str:
        if self.complete:
            return (
                f"COMPLETE: {self.attributed_bytes // MIB} MiB attributed over "
                f"{self.attributed_items} item(s)"
            )
        parts = [
            f"INCOMPLETE: {self.attributed_bytes // MIB} MiB of "
            f"{self.total_bytes // MIB} MiB attributed "
            f"({self.attributed_items}/{self.total_items} item(s)), "
            f"{self.missing_bytes // MIB} MiB unattributed"
        ]
        if self.starts_after_process_start:
            parts.append(
                "the record does not reach process start, so every post "
                "allocated during the boot is missing by construction; arm "
                f"{TRACE_ENV} on the server process and re-boot"
            )
        if self.note:
            parts.append(self.note)
        return ". ".join(parts)


@dataclasses.dataclass(frozen=True)
class SiteFootprint:
    """One allocating python line and what it is holding."""

    site: str
    bytes: int
    count: int
    stack: Tuple[str, ...] = ()

    @property
    def mib(self) -> int:
        return self.bytes // MIB


def _device_segments(snapshot: Mapping[str, Any]) -> List[dict]:
    return list(snapshot.get("segments") or ())


def resident_attribution(
    snapshot: Mapping[str, Any],
    *,
    include_inactive: bool = False,
) -> Tuple[List[SiteFootprint], Coverage]:
    """Per-callsite RESIDENT bytes, from ``segments[].blocks[].frames``.

    This is the structure that answers "what is the allocator holding right
    now, and who asked for it", because it is keyed on the blocks themselves
    rather than on an event history. Its one precondition is that recording was
    armed before the blocks were allocated -- which is exactly what
    :func:`arm_process_trace` at process start guarantees and what the #602
    ``/start_profile`` capture could not.

    ``include_inactive`` folds in blocks the allocator holds but has handed
    back; they are reserved bytes that the KV pool cannot have either, so they
    belong in a budget discussion, but they are separated by default because
    they are not a live claim by any post.
    """
    sites: Dict[str, List[Any]] = {}
    attributed = 0
    attributed_items = 0
    total = 0
    total_items = 0
    for segment in _device_segments(snapshot):
        for block in segment.get("blocks") or ():
            size = int(block.get("size", 0))
            state = str(block.get("state", ""))
            live = state.startswith("active")
            if not live and not include_inactive:
                # Still counted in the denominator: an unattributed inactive
                # block is a real hole in the explanation of reserved bytes.
                total += size
                total_items += 1
                continue
            total += size
            total_items += 1
            site = python_site(block.get("frames") or ())
            if site is None:
                continue
            attributed += size
            attributed_items += 1
            entry = sites.setdefault(site, [0, 0, python_stack(block.get("frames"))])
            entry[0] += size
            entry[1] += 1

    footprints = [
        SiteFootprint(site=site, bytes=v[0], count=v[1], stack=v[2])
        for site, v in sites.items()
    ]
    footprints.sort(key=lambda f: f.bytes, reverse=True)

    reserved_total = sum(
        int(s.get("total_size", 0)) for s in _device_segments(snapshot)
    )
    coverage = Coverage(
        attributed_bytes=attributed,
        total_bytes=max(total, reserved_total),
        attributed_items=attributed_items,
        total_items=total_items,
        # torch populates block frames only for blocks allocated after
        # recording began. Unframed blocks therefore ARE the proof that the
        # recording started late; no other signal is needed.
        starts_after_process_start=attributed_items < total_items,
        note=(
            "torch fills block frames only for blocks allocated after recording "
            "began; unframed blocks predate it"
            if attributed_items < total_items
            else ""
        ),
    )
    return footprints, coverage


def churn_attribution(
    snapshot: Mapping[str, Any],
    *,
    device: int = 0,
) -> Tuple[List[SiteFootprint], Coverage, dict]:
    """Per-callsite OUTSTANDING bytes over the trace window, from ``device_traces``.

    Answers a different question from :func:`resident_attribution`: not "what is
    held" but "what was allocated and not given back during the recorded
    window". Under steady-state serving most forwards REUSE cached blocks, so
    this is small by construction and must not be read as a footprint -- the
    #602 window measured 31 MiB of peak outstanding against 1389-2189 MiB of
    realized internal demand.

    The third return value carries the window's own statistics, including the
    two things a reader needs in order to know whether to believe it:

    ``orphan_free_bytes``   frees whose allocation is not in the window. Any
                            non-zero value means the window starts mid-process,
                            so resident posts are outside it.
    ``entries``             a ring that came back exactly full has WRAPPED and
                            silently dropped its oldest events.
    """
    traces = list(snapshot.get("device_traces") or ())
    trace = list(traces[device]) if device < len(traces) else []

    live: Dict[int, Tuple[int, Optional[str], Tuple[str, ...]]] = {}
    outstanding = 0
    peak_outstanding = 0
    orphan_frees = 0
    orphan_free_bytes = 0
    alloc_bytes = 0
    for event in trace:
        action = str(event.get("action", ""))
        if action == "alloc":
            size = int(event.get("size", 0))
            frames = event.get("frames") or ()
            live[int(event.get("addr", 0))] = (
                size,
                python_site(frames),
                python_stack(frames),
            )
            outstanding += size
            alloc_bytes += size
            peak_outstanding = max(peak_outstanding, outstanding)
        elif action == "free_completed":
            addr = int(event.get("addr", 0))
            entry = live.pop(addr, None)
            if entry is None:
                orphan_frees += 1
                orphan_free_bytes += int(event.get("size", 0))
            else:
                outstanding -= entry[0]

    sites: Dict[str, List[Any]] = {}
    unattributed = 0
    for size, site, stack in live.values():
        if site is None:
            unattributed += size
            continue
        entry = sites.setdefault(site, [0, 0, stack])
        entry[0] += size
        entry[1] += 1
    footprints = [
        SiteFootprint(site=site, bytes=v[0], count=v[1], stack=v[2])
        for site, v in sites.items()
    ]
    footprints.sort(key=lambda f: f.bytes, reverse=True)

    stats = {
        "entries": len(trace),
        "alloc_bytes": alloc_bytes,
        "outstanding_bytes": outstanding,
        "peak_outstanding_bytes": peak_outstanding,
        "orphan_frees": orphan_frees,
        "orphan_free_bytes": orphan_free_bytes,
        "window_seconds": (
            (int(trace[-1]["time_us"]) - int(trace[0]["time_us"])) / 1e6
            if len(trace) >= 2
            else 0.0
        ),
    }
    coverage = Coverage(
        attributed_bytes=outstanding - unattributed,
        total_bytes=outstanding,
        attributed_items=len(sites),
        total_items=len(live),
        starts_after_process_start=orphan_frees > 0,
        note=(
            f"{orphan_frees} free(s) in the window have no allocation in it, so "
            "the window begins mid-process and cannot contain a resident post"
            if orphan_frees
            else ""
        ),
    )
    return footprints, coverage, stats
