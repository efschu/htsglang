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
* ``device_traces`` is a fixed-size RING. ``max_entries`` is therefore not
  capped BY DEFAULT (torch's default is effectively unbounded): the #602
  capture came back with exactly 100000 entries, i.e. full, i.e. wrapped,
  retaining only the last ~10.7 s of a boot-long history. #1054b adds an
  OPT-IN cap (:data:`CAP_ENV`) for the one configuration that needs it -- a
  window held open through serving on a swapless box -- and the objection
  above is answered rather than overruled: what made the #602 wrap harmful was
  that it was SILENT, so a capped ring here reports its exact drop count
  (:func:`ring_loss_report`) with every dump, derived from torch's own
  monotone allocation counter. Uncapped remains the default and the better
  record.

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
    "hold_trace_through_serving",
    "ring_loss_report",
    "trace_armed",
    "dump_trace",
    "is_recording_phases",
    "mark",
    "mark_serving",
    "read_marks",
    "read_serving_marks",
    "reset_serving_pacer",
    "SERVING_PHASE",
    "SERVING_INTERVAL_ENV",
    "list_boots",
    "boot_id",
    "publish_boot_id",
    "dump_ledger",
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

#: #1054: HOLD THE WINDOW OPEN PAST ``boot_complete``, for the one question
#: the boot-bounded window cannot answer.
#:
#: ``disarm_process_trace`` closes the window at ``boot_complete`` because every
#: post this instrument was built to attribute is resident by then, and serving
#: would otherwise grow the ring for the life of the process. That reasoning is
#: correct for the RESIDENT question and wrong for the TRANSIENT one: boot 24
#: (2026-08-31) died of a CUDA OOM at ``fla/chunk_o.py:146
#: torch.zeros_like(v)`` inside the GDN extend kernel, minutes into serving,
#: with the corridor guard predicting a trough it did not reach. That allocation
#: happens only under real prefill depth, i.e. only AFTER the window this
#: instrument closes -- so the term stayed modeled while the ledger printed, in
#: 80 of 82 boots, that six of its terms are "neither measured nor bounded".
#:
#: Set this to keep recording through serving. It is a MEASUREMENT-RUN setting
#: and it says so in the log: the ring is uncapped by design (see
#: :func:`arm_process_trace`), so host RAM grows with every allocation the
#: process makes for as long as it runs. Never set it on an acceptance boot.
HOLD_ENV = "SGLANG_VRAM_FLIGHT_HOLD"

#: #1054b: CAP THE RING, AND SAY EXACTLY WHAT THE CAP COST.
#:
#: ``arm_process_trace`` deliberately passes no ``max_entries`` (see its
#: docstring): #602's capture came back exactly full at 100000, holding its
#: final 10.7 seconds, and a ring that wraps SILENTLY turns "this post has no
#: allocation event" into an untrue statement. That reasoning is about SILENCE,
#: not about caps. It stops applying the moment the wrap is COUNTED.
#:
#: And a cap is required once the window is held open through serving
#: (:data:`HOLD_ENV`): the ring then grows for the life of the process on a
#: swapless box that has already killed serving by host OOM without foreign
#: load, where serving carries ``oom_score_adj=500`` and is the preferred
#: victim. A diagnostic boot that kills the box measures nothing.
#:
#: So: set this to bound the ring, and the drop count is DERIVED rather than
#: guessed. ``torch.cuda.memory_stats()["allocation.all.allocated"]`` is a
#: monotone count of every allocation the process has made; sampled at arm time
#: and again at each dump, the difference is how many events the ring was
#: offered, and everything beyond the cap fell out of it. That number is
#: printed with every dump. An honest loss counter beside a bounded ring beats
#: a complete ring that ends the run -- and the ring is BYCATCH here anyway:
#: the load-bearing artifacts are the snapshots taken at the corridor guard's
#: first dip and in the crash handler, whose per-block stacks come from the
#: allocator's live segments, not from the trace ring.
CAP_ENV = "SGLANG_VRAM_FLIGHT_MAX_ENTRIES"

#: Carries the boot id from the launcher to every rank it spawns. Set by
#: :func:`publish_boot_id` and inherited through the environment, which is the
#: only channel the launcher and its children share before the ranks exist.
BOOT_ID_ENV = "SGLANG_VRAM_FLIGHT_BOOT_ID"

#: Directory for the phase-mark log (source 1) and any snapshot dumps. Absent,
#: every entry point in this module returns immediately and allocates nothing.
DIR_ENV = "SGLANG_VRAM_FLIGHT_DIR"

#: The boot ledger: one mark per boot POST, paired by name by its readers
#: (``reconcile`` asks for the ``weights_loaded -> kv_pool_sized`` delta).
MARKS_FILE_STEM = "flight_marks_rank"

#: The serving series (#684), kept in its OWN file for the reason above: a
#: boot post is a unique boundary and a serving sample is a time series, and
#: thousands of the latter in the ledger would turn a table of posts into a
#: log with posts in it. Same schema, same writer, different destination.
SERVING_FILE_STEM = "flight_serving_rank"

#: Phase name of a serving sample. Distinct from every boot post so a reader
#: can never mistake one for a boundary.
SERVING_PHASE = "serving"

#: Seconds between serving samples. The call site is once per scheduler
#: iteration -- thousands of times a second -- so the pacer is what makes the
#: instrument affordable. ``0`` disables the series while leaving the boot
#: marks armed.
SERVING_INTERVAL_ENV = "SGLANG_VRAM_FLIGHT_SERVING_S"
DEFAULT_SERVING_INTERVAL_S = 30.0

#: The boot boundaries, in order, and what the gap AFTER each one contains.
#: This is the contract between the instrument and its reader, and a
#: registered test asserts that the serving tree calls :func:`mark` with
#: exactly these names -- no more, so a typo cannot invent a phase, and no
#: fewer, so a deleted call site cannot quietly shrink the record. #602's
#: lesson: twelve green fixture tests passed while the production carrier
#: lacked the field they all built by hand.
BOOT_PHASES: Tuple[Tuple[str, str], ...] = (
    ("process_start", "before any CUDA allocation; the context is not yet bound"),
    (
        "nccl_init_begin",
        "before any process group is built; the communicator buffers this "
        "gap allocates are TERM_NCCL_BUFFERS, which #595 added to the ledger "
        "taxonomy and left with nowhere to be seen",
    ),
    (
        "nccl_init_end",
        "every process group of this launch now exists, the phase-flip "
        "secondary groups included",
    ),
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
    global _trace_cap, _alloc_count_at_arm
    cap = _configured_cap()
    try:
        import torch

        # enabled/context/stacks are left at their defaults on purpose: they
        # already are 'all'/'all'/'all'. Naming them would suggest that the
        # #602 capture lacked stacks because of a flag, which it did not.
        if cap:
            torch.cuda.memory._record_memory_history(max_entries=cap)
        else:
            torch.cuda.memory._record_memory_history()
        _alloc_count_at_arm = _allocations_so_far()
    except Exception as e:  # pragma: no cover - torch/platform differences
        logger.warning("VRAM flight recorder could not arm the trace: %s", e)
        return False
    _trace_armed = True
    _trace_cap = cap
    if cap:
        logger.info(
            "VRAM flight recorder: allocation recording armed for this process "
            "with a ring of %d entries (%s). BOUNDED ON PURPOSE and the cost is "
            "COUNTED, not hidden: every dump prints how many allocation events "
            "the ring was offered beyond this cap, derived from torch's own "
            "monotone allocation counter. Uncapped is the better record and a "
            "worse risk once the window is held through serving on a swapless "
            "box.",
            cap,
            CAP_ENV,
        )
    else:
        logger.info(
            "VRAM flight recorder: allocation recording armed for this process "
            "(uncapped ring). Host RAM grows with the number of allocations; "
            "this is a measurement boot, not a serving configuration."
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


_trace_cap: int = 0
_alloc_count_at_arm: Optional[int] = None


def _configured_cap() -> int:
    """The ring bound, or 0 for the uncapped default. Never negative."""
    raw = (os.environ.get(CAP_ENV) or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; leaving the ring uncapped", CAP_ENV, raw
        )
        return 0


def _allocations_so_far() -> Optional[int]:
    """torch's monotone count of allocations made by this process.

    The denominator of the drop count. ``None`` when the counter is
    unavailable, and a ``None`` is reported as UNKNOWN rather than as zero
    drops -- an unmeasurable loss is not an absent loss.
    """
    try:
        import torch

        stats = torch.cuda.memory_stats()
    except Exception:  # pragma: no cover - no CUDA / no context yet
        return None
    for key in ("allocation.all.allocated", "allocation.all.current"):
        if key in stats:
            return int(stats[key])
    return None


def ring_loss_report() -> str:
    """How much the capped ring dropped, in words, for the dump's log line."""
    if not _trace_cap:
        return "ring uncapped (no events dropped)"
    if _alloc_count_at_arm is None:
        return (
            f"ring capped at {_trace_cap} entries; DROPPED COUNT UNKNOWN "
            "(torch's allocation counter was unreadable at arm time -- treat "
            "this trace as possibly truncated, never as complete)"
        )
    now = _allocations_so_far()
    if now is None:
        return (
            f"ring capped at {_trace_cap} entries; DROPPED COUNT UNKNOWN "
            "(allocation counter unreadable now)"
        )
    offered = max(0, now - _alloc_count_at_arm)
    dropped = max(0, offered - _trace_cap)
    if dropped:
        return (
            f"ring capped at {_trace_cap} entries; {offered} allocation events "
            f"were offered since arming, so {dropped} FELL OUT of the ring. "
            "The trace holds only the most recent window; an absence in it is "
            "NOT evidence that a post never allocated. The per-block stacks in "
            "the segments below are unaffected -- they come from the live "
            "allocator, not from the ring."
        )
    return (
        f"ring capped at {_trace_cap} entries; {offered} events offered, none "
        "dropped -- this trace is COMPLETE for the window since arming"
    )


def hold_trace_through_serving() -> bool:
    """#1054: is this a measurement run that keeps recording past the boot?

    Read at the ONE disarm site rather than inside ``disarm_process_trace``,
    deliberately: the disarm verb must keep meaning "stop recording" for every
    other caller, including a future one that stops the trace on purpose. What
    is conditional is the boot's DECISION to stop, not the ability to.
    """
    return bool(os.environ.get(HOLD_ENV))


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
        # #1054b: the loss statement travels WITH the artifact. A reader who
        # opens this pickle six boots from now must not have to reconstruct
        # whether the ring wrapped -- the number is in the log beside the path.
        logger.info(
            "VRAM flight recorder: snapshot %s written to %s -- %s",
            tag,
            path,
            ring_loss_report(),
        )
        return path
    except Exception as e:  # pragma: no cover - torch/filesystem differences
        logger.warning("VRAM flight recorder could not dump snapshot %s: %s", tag, e)
        return None


# ---------------------------------------------------------------------------
# Source 1 + 3: phase marks
# ---------------------------------------------------------------------------


def is_recording_phases() -> bool:
    return bool(os.environ.get(DIR_ENV))


_boot_id: Optional[str] = None
_ledger_build_index: int = 0


def _self_identity() -> str:
    try:
        import psutil

        me = psutil.Process()
        return f"{me.pid}-{int(me.create_time())}"
    except Exception:  # pragma: no cover - psutil absence / permissions
        return f"{os.getpid()}-{int(time.time())}"


def publish_boot_id() -> str:
    """Fix this boot's id from the LAUNCHER and hand it to the ranks.

    THE LAUNCHER AND THE RANKS MUST AGREE, and before #605's first real boot
    they did not. A rank derives the id from its parent (the launcher), which
    is right; the launcher deriving it the same way would name ITS parent --
    the shell -- so the modeled ledger, which is built during argument
    resolution in the launcher, would have been filed under a different id
    than the marks. Reconciliation matches the two by id, so it would have
    found nothing even once the dump landed in the right place.

    Publishing through the environment removes the guesswork: children inherit
    it, and the value equals what the parent-based derivation produces, so a
    rank spawned by an older launcher still agrees.
    """
    global _boot_id
    existing = os.environ.get(BOOT_ID_ENV)
    if existing:
        _boot_id = existing
        return existing
    _boot_id = _self_identity()
    os.environ[BOOT_ID_ENV] = _boot_id
    return _boot_id


def boot_id() -> str:
    """An id shared by every process of ONE boot, distinct across boots.

    Read from :data:`BOOT_ID_ENV` when the launcher published one. Otherwise
    derived from this process's PARENT, because the callers that need it
    without a published id are the ranks, whose parent IS the launcher. A
    per-process uuid would tag each rank's file differently and make
    cross-rank reading impossible, which is the opposite of what the id is for.
    """
    global _boot_id
    if _boot_id is not None:
        return _boot_id
    published = os.environ.get(BOOT_ID_ENV)
    if published:
        _boot_id = published
        return _boot_id
    try:
        import psutil

        parent = psutil.Process().parent()
        _boot_id = f"{parent.pid}-{int(parent.create_time())}"
    except Exception:  # pragma: no cover - psutil absence / permissions
        _boot_id = f"self{os.getpid()}-{int(time.time())}"
    return _boot_id


def dump_ledger(ledgers) -> Optional[str]:
    """Write the MODELED per-card ledger beside the measured marks.

    Called from ``engine.build_card_ledgers``, i.e. wherever a boot builds a
    ledger at all, rather than from a caller on one reserve path. No-op unless
    the recorder is armed.

    Rebuilt ledgers OVERWRITE: argument resolution can construct the ledger
    several times while it derives the reserve, and the last construction is
    the one closest to what the boot runs. ``build_index`` records how many
    times it happened, so a reader can tell a single derivation from a loop.
    """
    global _ledger_build_index
    directory = os.environ.get(DIR_ENV)
    if not directory:
        return None
    _ledger_build_index += 1
    boot = publish_boot_id()
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"ledger_{boot}.json")
        payload = {
            "boot_id": boot,
            "build_index": _ledger_build_index,
            "pid": os.getpid(),
            "wall": time.time(),
            "cards": [x.to_json() for x in ledgers],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        return path
    except OSError as e:  # pragma: no cover - filesystem differences
        logger.warning("VRAM flight recorder could not write the ledger: %s", e)
        return None


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


def _kv_arena_view() -> Dict[str, Any]:
    """The KV VMM arena's own committed total, for the mark that is being taken.

    SOURCE 4, and the one that turns an argument into a chain. Sources 1-3
    could see that torch's ``reserved`` exceeded NVML's resident bytes by
    4.6-8.3 GiB per rank on the ship config, and could say WHY in prose -- the
    KV pool is a virtual-memory arena whose address space is reserved up front
    and whose physical pages are mapped incrementally -- but could not measure
    the split, because neither torch nor NVML knows where the arena's
    watermark sits. The arena does. Reading it here makes

        commit watermark -> resident bytes -> free VRAM -> corridor

    a measured chain instead of a plausible story.

    ``retained`` is reported separately from ``backed`` on purpose: parked
    physical handles are memory the arena still owns but has unmapped, so NVML
    charges the process for them while the backed total does not. Folding them
    together would recreate, one level down, exactly the false-zero this
    instrument was fixed for.

    Import is local and the whole body is guarded: a boot must not acquire a
    new way to fail by being measured, and a rank with no arena at all (the
    common case before the pool exists) simply contributes nothing.
    """
    try:
        from sglang.srt.mem_cache.kv_vmm_backing import arena_census

        census = arena_census()
    except Exception:  # pragma: no cover - the instrument never breaks a boot
        return {}
    if not census:
        return {}
    totals = {"reserved": 0, "backed": 0, "retained": 0, "arenas": 0}
    for row in census.values():
        for key in totals:
            totals[key] += int(row.get(key, 0) or 0)
    return {
        "kv_arena_reserved_bytes": totals["reserved"],
        "kv_arena_backed_bytes": totals["backed"],
        "kv_arena_retained_bytes": totals["retained"],
        "kv_arena_count": totals["arenas"],
    }


def mark(
    phase: str,
    *,
    rank: int = 0,
    device_index: Optional[int] = None,
    directory: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    _filename: Optional[str] = None,
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
        # Identifies the boot this mark belongs to. Without it a file that
        # accumulates boots cannot be read as one boot, and the delta across
        # the seam is a number describing nothing.
        "boot_id": boot_id(),
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
        #
        # A LARGE negative is a different animal and may not be floored away.
        # On the ship config torch reports up to 7162 MiB MORE reserved than
        # NVML says this process holds -- reservation that carries no physical
        # backing. Flooring that to 0 published "this rank has no CUDA context,
        # no NCCL buffer and no JIT workspace on this card" for a rank whose
        # context alone measured 886 MiB. The floor was built for quantisation
        # and was swallowing gigabytes, so the two cases are now separated and
        # each is named:
        #
        #   non_torch_measurable  -- False when torch's books exceed the card's,
        #                            in which case non_torch_bytes is NOT a
        #                            measurement of the residue and a reader
        #                            must not treat its 0 as an observation.
        #   unbacked_reservation  -- how far reserved exceeds resident. This is
        #                            the measurement that replaces the lost one.
        residue = int(record["nvml_self_bytes"]) - int(record["reserved_bytes"])
        record["non_torch_bytes"] = max(0, residue)
        record["unbacked_reservation_bytes"] = max(0, -residue)
        record["non_torch_measurable"] = residue >= 0
    if "reserved_peak_bytes" in record and "reserved_bytes" in record:
        # #612. The allocator peak ABOVE what is still resident at this mark:
        # the transient the ledger's TERM_LOAD_TRANSIENT stands for. Computed
        # here, at write time, for the same reason non_torch_bytes is -- the
        # subtraction is the measurement, and a reader of one line should not
        # have to know which two counters to subtract. RESERVED and not
        # ALLOCATED on purpose: NVML sees the allocator's reservation, so the
        # reserved pair is the one that moves the free-memory floor the
        # corridor is measured against.
        record["allocator_transient_bytes"] = max(
            0, int(record["reserved_peak_bytes"]) - int(record["reserved_bytes"])
        )
    record.update(_kv_arena_view())
    if extra:
        record["extra"] = dict(extra)
    try:
        os.makedirs(directory, exist_ok=True)
        # ONE RECORD BUILDER, TWO DESTINATIONS (#684). The serving series is
        # written by this same function so the two can never drift into two
        # schemas -- a copy of the record layout is exactly how the field a
        # post-mortem needs ends up present in one file and missing in the
        # other.
        path = os.path.join(
            directory, _filename or f"{MARKS_FILE_STEM}{rank}.jsonl"
        )
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:  # pragma: no cover - filesystem differences
        logger.warning("VRAM flight recorder could not write a phase mark: %s", e)
        return record
    return record


class _ServingPacer:
    """Decides when the next serving sample is due, on the MONOTONIC clock.

    Wall time is not usable for this: an NTP step backwards would stall the
    series for the length of the step, and a step forwards would flood it --
    and the one boot that most needs the record is the one whose clock is
    being corrected because it has been up a long time.

    The interval is resolved ONCE and cached, on the same argument the
    corridor sampler's ``start`` uses: this is consulted thousands of times a
    second, and an env lookup per scheduler iteration is a cost the instrument
    has no reason to charge.
    """

    def __init__(self) -> None:
        self._last: Optional[float] = None
        self._interval: Optional[float] = None

    def interval_s(self) -> float:
        if self._interval is None:
            raw = os.environ.get(SERVING_INTERVAL_ENV)
            if raw is None:
                self._interval = DEFAULT_SERVING_INTERVAL_S
            else:
                try:
                    self._interval = float(raw)
                except (TypeError, ValueError):
                    # An unreadable cadence must not silence the instrument:
                    # a typo in a boot script would otherwise cost the next
                    # post-mortem its whole serving record.
                    logger.warning(
                        "VRAM flight recorder: %s=%r is not a number; using "
                        "the %.0fs default",
                        SERVING_INTERVAL_ENV,
                        raw,
                        DEFAULT_SERVING_INTERVAL_S,
                    )
                    self._interval = DEFAULT_SERVING_INTERVAL_S
        return self._interval

    def due(self, now: float) -> bool:
        interval = self.interval_s()
        if interval <= 0:
            return False
        # The FIRST call always marks, so a boot has a serving datum before
        # any drift starts and the series has a baseline to be read against.
        if self._last is None or (now - self._last) >= interval:
            self._last = now
            return True
        return False

    def reset(self) -> None:
        self._last = None
        self._interval = None


_serving_pacer = _ServingPacer()


def reset_serving_pacer() -> None:
    """Forget the cadence and the last sample time (tests, and re-arming)."""
    _serving_pacer.reset()


def mark_serving(
    *,
    rank: int = 0,
    device_index: Optional[int] = None,
    directory: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    now: Optional[float] = None,
) -> Optional[dict]:
    """One paced serving-time sample. No-op unless :data:`DIR_ENV` is set.

    WHY THIS EXISTS, MEASURED. The recorder's last boot post is
    ``first_forward``. On 2026-08-16 an instance died 36 minutes later with
    ``76.38 MiB is free ... Process 1920108 has 4.29 GiB memory in use``, and
    naming that process took hours of log archaeology plus a pid-clock
    interpolation across two boots. Every fact needed to answer it in one line
    -- the card's free bytes and the full pid->bytes map of everyone on it --
    was already computed by :func:`_nvml_view` on every mark. Nothing was
    marking.

    NOT THE SAME JOB AS THE #605 CORRIDOR SAMPLER, which does run during
    serving: that one keeps a fixed-size RAM ring, so it dies with the process
    that crashes, and its ``Sample`` discards the per-pid map it reads. Marks
    are appended to a FILE and survive the crash -- which is not theoretical,
    since the surviving boot marks are what made that pid clock calibratable
    after the fact.

    Returns the record when a sample was taken, ``None`` when the pace was not
    due. Never raises: the caller is a scheduler iteration.
    """
    directory = directory or os.environ.get(DIR_ENV)
    if not directory:
        return None
    if not _serving_pacer.due(time.monotonic() if now is None else float(now)):
        return None
    return mark(
        SERVING_PHASE,
        rank=rank,
        device_index=device_index,
        directory=directory,
        extra=extra,
        _filename=f"{SERVING_FILE_STEM}{rank}.jsonl",
    )


def read_serving_marks(
    directory: str, *, boot: Optional[str] = None
) -> Dict[int, List[dict]]:
    """The serving series for ONE boot, ``{pid: [sample, ...]}`` (#684).

    Same grouping and boot-selection contract as :func:`read_marks`, over the
    serving file instead of the boot ledger.
    """
    return read_marks(directory, boot=boot, _stem=SERVING_FILE_STEM)


def read_marks(
    directory: str,
    *,
    boot: Optional[str] = None,
    _stem: str = MARKS_FILE_STEM,
) -> Dict[int, List[dict]]:
    """``{pid: [mark, ...]}`` for ONE boot, latest by default.

    THE FILE HOLDS MORE THAN ONE BOOT and that is the point of the format: a
    rank appends, so a crashed boot keeps the boundaries it did reach and the
    next boot does not erase them. It also means a reader that returns the
    whole file returns marks from several process lifetimes, and
    :func:`phase_deltas` over that produces a delta across the seam --
    ``boot_complete`` of one boot to ``process_start`` of the next -- which is
    a number describing nothing, printed with the same confidence as a real
    post. Selecting a boot is therefore not a convenience, it is the thing that
    keeps the output honest once the recorder is armed on every boot.

    GROUPED BY PID, NOT BY RANK, and not by the file the marks were found in.
    Marks are FILED under the TP rank, and the ship config runs ``--tp-size 1
    --pp-size 3``: the TP rank is 0 in all three processes, so all three append
    to ``flight_marks_rank0.jsonl``. Keying the result on the rank field
    therefore merged three processes on three different cards into a single
    timeline, and differencing it produced posts that no card ever paid --
    a 20480 MiB card's CUDA context billed to the 32607 MiB card. The pid is
    stamped by the process that took the mark and cannot collide while that
    process lives, so the pid is the grouping key. Callers that want the rank
    read it off any mark in the group.

    THE BOOT IS RESOLVED ACROSS ALL FILES, not per file. Ranks do not stop
    writing at the same instant, so "the last boot_id in this file" can name a
    different boot for each rank, which silently returns a mixture.

    Pass ``boot="all"`` to get every record, for a reader that means to handle
    the seams itself.
    """
    out: Dict[int, List[dict]] = {}
    if not os.path.isdir(directory):
        return out
    records: List[dict] = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith(_stem) and name.endswith(".jsonl")):
            continue
        path = os.path.join(directory, name)
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
    if not records:
        return out
    if boot != "all":
        if boot is not None:
            wanted = boot
        else:
            latest = max(records, key=lambda r: r.get("wall") or 0.0)
            wanted = latest.get("boot_id")
        selected = [r for r in records if r.get("boot_id") == wanted]
        # Records written before boot_id existed carry none; keeping them out
        # of a boot-scoped read is right, but silently returning nothing would
        # look like "the boot wrote no marks".
        if not selected and wanted is None:
            logger.warning(
                "flight recorder: %s carries marks with no boot_id; they "
                "predate boot-scoped reads and are being returned whole",
                directory,
            )
            selected = records
        records = selected
    for record in records:
        out.setdefault(int(record.get("pid", 0)), []).append(record)
    for group in out.values():
        group.sort(key=lambda r: r.get("monotonic") or 0.0)
    return out


def list_boots(directory: str) -> List[Tuple[str, float, int]]:
    """``[(boot_id, first wall clock, mark count), ...]``, oldest first."""
    seen: Dict[str, List[Any]] = {}
    for records in read_marks(directory, boot="all").values():
        for record in records:
            boot_id = record.get("boot_id")
            if boot_id is None:
                continue
            entry = seen.setdefault(boot_id, [float(record.get("wall", 0.0)), 0])
            entry[0] = min(entry[0], float(record.get("wall", 0.0)))
            entry[1] += 1
    return sorted(
        ((boot_id, v[0], v[1]) for boot_id, v in seen.items()), key=lambda t: t[1]
    )


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
        if a.get("boot_id") != b.get("boot_id"):
            # The seam between two boots. Skipped rather than reported: the
            # difference between one boot's last mark and the next boot's
            # first is not a post, and printing it beside real posts in the
            # same table is exactly how a reader would come to trust it.
            logger.warning(
                "flight recorder: skipping the delta from %s to %s, which "
                "spans two boots (%s -> %s)",
                a.get("phase"),
                b.get("phase"),
                a.get("boot_id"),
                b.get("boot_id"),
            )
            continue
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
