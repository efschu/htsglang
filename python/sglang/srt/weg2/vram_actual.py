"""VRAM-Vertrag M2: the rank writes its VRAM ACTUAL per PID and per category
into its RankState (block ``vram``), not into a log line.

User order 28.09. ~20:45Z ("über logfiles?"): control values travel as
versioned records; 29.09. ("die VRAM-Belegung sollte doch schon vorher gelöst
sein -- per Profil"): the planner plans, the ranks REPORT what they hold.
Design: /spinning/gpu-arb/docs/VRAM-VERTRAG-0929.md §3.4 (categories, who
writes), §3.7 (the block), §4 M2. This module measures and reports; it never
sizes a pool and never decides anything (M6 binds the plan, M7 raises events).

THE BLOCK (``weg2.rank_vram/1``, all figures MiB, integers)::

    {"schema": "weg2.rank_vram/1", "plan_id": null, "contract": {},
     "actual": {<12 categories>, "unattributed": n},   # sum == nvml_pid_mib
     "unattributed_parts": {...}, "findings": [...],
     "nvml_pid_mib": n, "torch_reserved_mib": n, "torch_allocated_mib": n,
     "foreign_mib": n, "partner_mib": n,
     "peak_by_state": {"n=3,S0,extend=0": {"reserved_peak": n,
                        "allocated_peak": n, "transient": n, "nvml_pid": n, "n": k}},
     "marks": {"weights_loaded": {"nvml_pid": n, "reserved": n,
               "allocated": n, "cost_us": n, "write_us": n, "n": k}},
     "sources": {<category>: "<where the figure came from>"},
     "mark": "<last mark>", "ts": t, "seq": k}

INVARIANT. ``sum(actual.values()) == nvml_pid_mib`` exactly: ``unattributed``
is the remainder, never a plug. It can be NEGATIVE: torch reserved bytes that
carry no physical backing (measured: D TP0 z30 ``other -404``). A remainder
beyond :data:`UNATTRIBUTED_NAMED_MIB` is a FINDING and names its parts
(``torch_untagged_allocated``, ``non_torch_growth``, ``tag:<unknown>``) --
the rc12c "untagged ~2 GiB" gap may never again be one silent number.

``foreign`` is NOT in ``actual``: it is not this PID's memory, and the
invariant is a statement about this PID. It is reported beside it as
``foreign_mib`` (NVML compute processes on this card that belong to no group
of this boot) and ``partner_mib`` (the other group of this boot on this card:
the source of the plan's ``asleep``), agreed with the planner seat 29.09.

CATEGORY SOURCES (all pre-existing instruments, §3.4):
  weights          TMS region tags ``weights``/``weights_<chunk>`` resident
                   (the saver's own ``tms_tag_bytes``; paused = the weight
                   updater's ``offload_tags``), minus the expert rows they hold
  draft            tag ``weights_draft`` resident
  experts_resident R rows x row bytes of every MoEExpertOffloadCache buffer
                   (band tags ``weights_<n>_e<b>`` when #134 bands are on)
  experts_lru      the remaining (C scratch/LRU/staging + X seat) rows
  kv               tag ``kv_cache`` minus state pools; untagged pool bytes
                   without TMS; + the KV VMM arena's backed bytes (non-torch)
  state_pools      mamba/GDN pool ``mem_usage`` (inside the kv_cache tag)
  activation       0 in a snapshot (freed by the mark); the transient lives in
                   ``peak_by_state`` (WEG2-VRAM-PEAK windows, peak - start)
  graphs           tag ``cuda_graph`` resident
  allocator_slack  torch reserved - allocated
  cuda_ctx         NVML(pid) - torch reserved at the FIRST ``pre_weight_load``
                   mark (context + NCCL, fixed per driver/card), frozen
  staging, vision  no in-rank source yet -> 0, named in ``sources``

MARKS: the existing ``flight_recorder.mark`` calls (pre_weight_load,
weights_loaded, kv_pool_sized, capture_end, boot_complete) forward here even
when the flight recorder itself is unarmed; every flip leg
(``vram_peak_window.flip_leg``); the end of every WEG2-VRAM-PEAK window, where
only the in-memory max per state key is updated -- the NVML read and the
file write happen only when a max CHANGED (never once per decode round).

TRANSPORT: the attach record (``cache_controller.weg2_publish_rank_state``)
carries the block; every later change rewrites the same file atomically
(tmp + fsync + rename, ``rank_state.write_rank_state``) with ``vram.seq + 1``.
Before the attach the block is held in memory only.

HAND-SET NUMBERS (Planer-Schuld, named): :data:`UNATTRIBUTED_NAMED_MIB`
(256, the M2 metal marker) and :data:`INVARIANT_TOL_MIB` (64, §3.11 point 2).
Both are acceptance thresholds of the contract, not VRAM budgets; M6 takes
them from the plan.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace as _replace_dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import msgspec
from msgspec.structs import replace

logger = logging.getLogger(__name__)

VRAM_SCHEMA = "weg2.rank_vram/1"
MIB = 1 << 20

#: The own-PID categories of §3.4, in the order of the doc. ``foreign`` is
#: reported beside them (see module docstring), never inside.
CATEGORIES: Tuple[str, ...] = (
    "weights",
    "draft",
    "experts_resident",
    "experts_lru",
    "kv",
    "state_pools",
    "activation",
    "graphs",
    "allocator_slack",
    "cuda_ctx",
    "staging",
    "vision",
)
UNATTRIBUTED = "unattributed"

#: A remainder beyond this is a finding with named parts (M2 metal marker
#: "unattributed < 256 MiB je Rang"). Planer-Schuld: M6 takes it from the plan.
UNATTRIBUTED_NAMED_MIB = 256
#: §3.11 point 2 (64 MiB, the front's slack size): the tolerance a reader
#: applies to ``sum(actual) == nvml_pid_mib`` across rounding. Planer-Schuld.
INVARIANT_TOL_MIB = 64

_BLOCK_KEYS = frozenset({
    "schema", "plan_id", "contract", "actual", "unattributed_parts", "findings",
    "nvml_pid_mib", "torch_reserved_mib", "torch_allocated_mib", "foreign_mib",
    "partner_mib", "peak_by_state", "marks", "sources", "mark", "ts", "seq",
})
_PEAK_KEYS = frozenset({"reserved_peak", "allocated_peak", "transient", "nvml_pid", "n"})
_MARK_KEYS = frozenset({"nvml_pid", "reserved", "allocated", "cost_us", "write_us", "n"})


class VramBlockSchemaError(ValueError):
    """A ``vram`` block of another schema, or with a field this reader does
    not know. Raised by name, never guessed around (IPC plan §2)."""


def _mib(b: Optional[int]) -> Optional[int]:
    return None if b is None else int(round(int(b) / MIB))


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


class VramBlock(msgspec.Struct, frozen=True, kw_only=True):
    """What one rank HOLDS, per category, as of its last mark."""

    actual: Dict[str, int]
    nvml_pid_mib: Optional[int]
    torch_reserved_mib: int
    torch_allocated_mib: int
    unattributed_parts: Dict[str, int] = {}
    findings: List[str] = []
    foreign_mib: Optional[int] = None
    partner_mib: Optional[int] = None
    peak_by_state: Dict[str, Dict[str, int]] = {}
    marks: Dict[str, Dict[str, int]] = {}
    sources: Dict[str, str] = {}
    mark: str = ""
    #: M1/M6: the plan this rank allocates against; null until M6 binds it.
    plan_id: Optional[str] = None
    #: M6: the planned MiB per category; empty until then.
    contract: Dict[str, int] = {}
    ts: float = 0.0
    seq: int = 0
    schema: str = VRAM_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "plan_id": self.plan_id, "contract": dict(self.contract),
            "actual": dict(self.actual), "unattributed_parts": dict(self.unattributed_parts),
            "findings": list(self.findings), "nvml_pid_mib": self.nvml_pid_mib,
            "torch_reserved_mib": self.torch_reserved_mib,
            "torch_allocated_mib": self.torch_allocated_mib,
            "foreign_mib": self.foreign_mib, "partner_mib": self.partner_mib,
            "peak_by_state": {k: dict(v) for k, v in self.peak_by_state.items()},
            "marks": {k: dict(v) for k, v in self.marks.items()},
            "sources": dict(self.sources), "mark": self.mark,
            "ts": self.ts, "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "VramBlock":
        if not isinstance(d, dict):
            raise VramBlockSchemaError("vram block is not an object")
        if d.get("schema") != VRAM_SCHEMA:
            raise VramBlockSchemaError(
                f"vram block schema {d.get('schema')!r}, this reader knows {VRAM_SCHEMA!r}")
        unknown = sorted(set(d) - _BLOCK_KEYS)
        if unknown:
            raise VramBlockSchemaError(f"vram block {VRAM_SCHEMA} has no field(s) {unknown}")
        actual = d.get("actual")
        if not isinstance(actual, dict):
            raise VramBlockSchemaError("vram block: actual is not an object")
        bad = sorted(set(actual) - set(CATEGORIES) - {UNATTRIBUTED})
        if bad:
            raise VramBlockSchemaError(f"vram block {VRAM_SCHEMA}: unknown categor(y/ies) {bad}")
        missing = sorted((set(CATEGORIES) | {UNATTRIBUTED}) - set(actual))
        if missing:
            raise VramBlockSchemaError(f"vram block {VRAM_SCHEMA}: actual lacks {missing}")
        for key, sub, allowed in (("peak_by_state", d.get("peak_by_state") or {}, _PEAK_KEYS),
                                  ("marks", d.get("marks") or {}, _MARK_KEYS)):
            for name, row in sub.items():
                extra = sorted(set(row) - allowed) if isinstance(row, dict) else ["<not an object>"]
                if extra:
                    raise VramBlockSchemaError(f"vram block {key}[{name!r}] has no field(s) {extra}")
        try:
            return cls(
                actual={k: int(v) for k, v in actual.items()},
                nvml_pid_mib=None if d.get("nvml_pid_mib") is None else int(d["nvml_pid_mib"]),
                torch_reserved_mib=int(d["torch_reserved_mib"]),
                torch_allocated_mib=int(d["torch_allocated_mib"]),
                unattributed_parts={k: int(v) for k, v in (d.get("unattributed_parts") or {}).items()},
                findings=[str(x) for x in (d.get("findings") or [])],
                foreign_mib=None if d.get("foreign_mib") is None else int(d["foreign_mib"]),
                partner_mib=None if d.get("partner_mib") is None else int(d["partner_mib"]),
                peak_by_state={k: {kk: int(vv) for kk, vv in v.items()}
                               for k, v in (d.get("peak_by_state") or {}).items()},
                marks={k: {kk: int(vv) for kk, vv in v.items()} for k, v in (d.get("marks") or {}).items()},
                sources={k: str(v) for k, v in (d.get("sources") or {}).items()},
                mark=str(d.get("mark") or ""),
                plan_id=d.get("plan_id"),
                contract={k: int(v) for k, v in (d.get("contract") or {}).items()},
                ts=float(d.get("ts") or 0.0),
                seq=int(d.get("seq") or 0),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise VramBlockSchemaError(f"vram block incomplete or mistyped: {e}") from e

    def invariant_gap_mib(self) -> Optional[int]:
        """``sum(actual) - nvml_pid_mib`` (0 by construction), None without NVML."""
        if self.nvml_pid_mib is None:
            return None
        return int(sum(self.actual.values())) - int(self.nvml_pid_mib)

    def line(self) -> str:
        """One human line (display only; no reader may parse it)."""
        top = sorted(((v, k) for k, v in self.actual.items() if k != UNATTRIBUTED and v), reverse=True)
        return (
            f"nvml_pid={self.nvml_pid_mib} unattributed={self.actual.get(UNATTRIBUTED)} "
            f"reserved={self.torch_reserved_mib} allocated={self.torch_allocated_mib} "
            f"partner={self.partner_mib} foreign={self.foreign_mib} "
            + " ".join(f"{k}={v}" for v, k in top)
            + f" states={len(self.peak_by_state)} seq={self.seq} mark={self.mark}"
            + (f" FINDINGS: {'; '.join(self.findings)}" if self.findings else "")
        )


# ---------------------------------------------------------------------------
# Attribution (pure)
# ---------------------------------------------------------------------------


class Reading(msgspec.Struct, frozen=True, kw_only=True):
    """Every figure one mark reads, in BYTES. Pure input of :func:`attribute`."""

    nvml_pid: Optional[int]
    reserved: int
    allocated: int
    #: device bytes per TMS region tag (resident AND paused -- the saver's own sum)
    tags: Dict[str, int] = {}
    #: tags the saver has currently paused (unmapped, NOT in nvml_pid)
    paused: FrozenSet[str] = frozenset()
    #: expert buffer rows: resident R rows / the rest (LRU + staging + seat)
    experts_resident: int = 0
    experts_lru: int = 0
    #: mamba/GDN state pools (inside the kv_cache region under TMS)
    state_pools: int = 0
    #: the KV pool's own size, used only when no kv_cache tag holds it
    kv_pool: int = 0
    #: KV VMM arena's physically backed bytes (cuMemMap, outside torch)
    kv_arena_backed: int = 0
    #: NVML(pid) - reserved at the first pre_weight_load mark
    ctx_baseline: Optional[int] = None
    #: parameter + buffer bytes on the device of the target / the draft model:
    #: the weights when no saver tag holds them (TMS off; NF's resident drafter)
    target_params: int = 0
    draft_params: int = 0


def category_of_tag(tag: str) -> Optional[str]:
    """The category a TMS region tag's resident bytes belong to, or None for
    a tag this reader does not know (then it is a named remainder part)."""
    t = str(tag)
    if t == "weights_draft":
        return "draft"
    if t == "kv_cache":
        return "kv"
    if t == "cuda_graph":
        return "graphs"
    if t == "weights":
        return "weights"
    if is_expert_band_tag(t):
        return "experts_band"
    if t.startswith("weights_"):
        rest = t[len("weights_"):]
        if rest.isdigit():
            return "weights"
    return None


def is_expert_band_tag(tag: str) -> bool:
    """``weights_<n>_e<b>`` (#134 band tags): the experts of one layer chunk."""
    parts = str(tag).split("_")
    return (len(parts) == 3 and parts[0] == "weights" and parts[1].isdigit()
            and parts[2].startswith("e") and parts[2][1:].isdigit())


def attribute(r: Reading) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str], List[str]]:
    """``(actual MiB, unattributed_parts MiB, sources, findings)`` of one reading.

    The decomposition is the measured one of WEG2-DC-BREAKDOWN
    (weg2_memory_saver.dc_breakdown, boot weg2xsn206): torch counts the
    saver's regions -- resident AND paused -- as reserved, so

        nvml_pid = resident tags + torch untagged (reserved - all tags) + non-torch

    and every category is carved out of exactly one of those three terms."""
    b: Dict[str, int] = {c: 0 for c in CATEGORIES}
    parts: Dict[str, int] = {}
    src: Dict[str, str] = {}
    tags = {str(t): int(v or 0) for t, v in r.tags.items() if int(v or 0) > 0}
    paused = {str(t) for t in r.paused}
    all_tags = sum(tags.values())
    paused_bytes = sum(v for t, v in tags.items() if t in paused)
    bands_resident = 0
    bands_total = 0
    # 1. the saver's regions
    for t, v in sorted(tags.items()):
        cat = category_of_tag(t)
        if cat == "experts_band":
            bands_total += v
            if t not in paused:
                bands_resident += v
            continue
        if t in paused:
            continue
        if cat is None:
            parts[f"tag:{t}"] = parts.get(f"tag:{t}", 0) + v
            continue
        b[cat] += v
    if tags:
        src["weights"] = "tms_tag_bytes(weights*) resident"
        src["draft"] = "tms_tag_bytes(weights_draft) resident"
        src["graphs"] = "tms_tag_bytes(cuda_graph) resident"
    # 1b. state pools live inside the kv_cache region (the pool init runs
    # under it); without a kv tag they are untagged torch allocations below
    kv_tagged = tags.get("kv_cache", 0) > 0
    if kv_tagged and "kv_cache" not in paused and r.state_pools:
        take = min(int(r.state_pools), b["kv"])
        b["kv"] -= take
        b["state_pools"] += take
        src["state_pools"] = "mamba_pool.mem_usage carved out of tag kv_cache"
    # 1c. experts: band tags carry them; else they sit in the weights tags
    # (the buffers are allocated under the layer's chunk region); else untagged
    exp_total = int(r.experts_resident) + int(r.experts_lru)
    exp_untagged = 0
    if bands_total:
        share = (r.experts_resident / exp_total) if exp_total else 1.0
        b["experts_resident"] += int(round(bands_resident * share))
        b["experts_lru"] += bands_resident - int(round(bands_resident * share))
        src["experts_resident"] = src["experts_lru"] = "tms band tags weights_<n>_e<b>, split by pool rows"
    elif exp_total:
        weights_family_total = sum(v for t, v in tags.items() if category_of_tag(t) == "weights")
        if b["weights"] >= exp_total:
            b["weights"] -= exp_total
            b["experts_resident"] += int(r.experts_resident)
            b["experts_lru"] += int(r.experts_lru)
            src["experts_resident"] = src["experts_lru"] = "expert buffer rows, carved out of weights tags"
        elif weights_family_total >= exp_total:
            # the weights tags are paused: the rows inside them are unmapped
            src["experts_resident"] = src["experts_lru"] = "expert buffer rows inside PAUSED weights tags (0 resident)"
        else:
            exp_untagged = exp_total
            src["experts_resident"] = src["experts_lru"] = "expert buffer rows, untagged torch allocations"
    # 2. torch, outside every region
    untagged_alloc = max(0, int(r.allocated) - all_tags)
    b["allocator_slack"] = max(0, int(r.reserved) - int(r.allocated))
    src["allocator_slack"] = "torch reserved - allocated"
    rest_alloc = untagged_alloc
    carve: List[Tuple[str, int, str]] = []
    if exp_untagged:
        carve += [("experts_resident", int(r.experts_resident), ""), ("experts_lru", int(r.experts_lru), "")]
    if r.draft_params and not tags.get("weights_draft"):
        carve.append(("draft", int(r.draft_params), "draft model parameter bytes (untagged)"))
    if r.target_params and not any(category_of_tag(t) in ("weights", "experts_band") for t in tags):
        carve.append(("weights", max(0, int(r.target_params) - exp_untagged),
                      "target model parameter bytes (untagged, no saver tag)"))
    if not kv_tagged:
        if r.kv_pool:
            carve.append(("kv", int(r.kv_pool), "kv pool size (untagged)"))
        if r.state_pools:
            carve.append(("state_pools", int(r.state_pools), "mamba_pool.mem_usage (untagged)"))
    for cat, want, why in carve:
        take = min(max(0, want), rest_alloc)
        b[cat] += take
        rest_alloc -= take
        if why:
            src[cat] = why
        if take < want:
            parts[f"short:{cat}"] = parts.get(f"short:{cat}", 0) - (want - take)
    if rest_alloc:
        parts["torch_untagged_allocated"] = rest_alloc
    # 3. what the driver charges this PID beyond torch's mapped books
    findings: List[str] = []
    if r.nvml_pid is not None:
        torch_mapped = int(r.reserved) - paused_bytes
        non_torch = int(r.nvml_pid) - torch_mapped
        arena = max(0, int(r.kv_arena_backed))
        if arena:
            take = min(arena, max(0, non_torch))
            b["kv"] += take
            non_torch -= take
            src["kv"] = src.get("kv", "tms_tag_bytes(kv_cache) resident") + " + kv VMM arena backed"
        if r.ctx_baseline is not None:
            b["cuda_ctx"] = max(0, min(max(0, non_torch), int(r.ctx_baseline)))
            src["cuda_ctx"] = "nvml_pid - torch reserved at first pre_weight_load (frozen)"
            growth = non_torch - b["cuda_ctx"]
        else:
            growth = non_torch
            src["cuda_ctx"] = "no pre_weight_load baseline in this process"
        if growth > 0:
            parts["non_torch_growth"] = growth
        elif growth < 0:
            # torch reserved more than the driver backs: an unbacked reservation
            parts["unbacked_reservation"] = growth
    else:
        findings.append("nvml_pid unreadable: no invariant, no remainder")
    src.setdefault("kv", "tms_tag_bytes(kv_cache) resident" if kv_tagged else "none")
    src.setdefault("state_pools", "none")
    src["activation"] = "0 in a snapshot; transient per state in peak_by_state"
    src["staging"] = "no in-rank source (M2 gap)"
    src["vision"] = "no in-rank source (M2 gap)"
    actual = {c: _mib(v) for c, v in b.items()}
    if r.nvml_pid is not None:
        actual[UNATTRIBUTED] = _mib(r.nvml_pid) - sum(actual.values())
    else:
        actual[UNATTRIBUTED] = 0
    parts_mib = {k: _mib(v) for k, v in parts.items() if _mib(v)}
    if abs(actual[UNATTRIBUTED]) > UNATTRIBUTED_NAMED_MIB:
        named = ", ".join(f"{k}={v}" for k, v in sorted(parts_mib.items(), key=lambda kv: -abs(kv[1])))
        findings.append(
            f"unattributed {actual[UNATTRIBUTED]} MiB beyond {UNATTRIBUTED_NAMED_MIB}: {named or 'no part named'}")
    return actual, parts_mib, src, findings


# ---------------------------------------------------------------------------
# State keys (agreed with the planner seat 29.09.: plan cells / flip_legs)
# ---------------------------------------------------------------------------


def pow2_bucket(x: int) -> int:
    """0 stays 0; otherwise the next power of two >= x."""
    x = int(x)
    if x <= 0:
        return 0
    return 1 << (x - 1).bit_length()


def d_state_key(n: int, stage: Optional[int], extend_rows: int) -> str:
    return f"n={int(n)},S{'-' if stage is None else int(stage)},extend={pow2_bucket(extend_rows)}"


def p_state_key(chunk_rows: int, stau: Optional[int]) -> str:
    return f"chunk={pow2_bucket(chunk_rows)},stau={'-' if stau is None else int(stau)}"


def flip_state_key(group: str, leg: str) -> str:
    """``flip=P->D,leg=sleep`` etc.: P sleeps on P->D and wakes on D->P; D the
    other way round (plan.flip_legs["P->D"])."""
    sleep = leg == "release"
    g = (group or "").upper()
    if g == "P":
        direction = "P->D" if sleep else "D->P"
    elif g == "D":
        direction = "D->P" if sleep else "P->D"
    else:
        direction = "?"
    return f"flip={direction},leg={'sleep' if sleep else 'wake'}"


# ---------------------------------------------------------------------------
# The in-rank recorder (one per process)
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """SGLANG_WEG2_VRAM_ACTUAL, read once per process (reset() re-reads)."""
    v = _REC.switch
    if v is None:
        try:
            from sglang.srt.environ import envs

            v = bool(envs.SGLANG_WEG2_VRAM_ACTUAL.get())
        except Exception:  # noqa: BLE001 -- an instrument never breaks a boot
            v = False
        _REC.switch = v
    return v


class _Nvml:
    """One cached NVML handle for this rank's card: init once, per mark only
    ``nvmlDeviceGetComputeRunningProcesses`` (the per-PID instrument of
    WEG2-DC / DC-BREAKDOWN). ``registry.nvml`` inits and shuts NVML per call
    and walks every card, which is the cost the < 1 ms target forbids."""

    def __init__(self) -> None:
        self.handle = None
        self.pynvml = None
        self.failed: Optional[str] = None

    def procs(self) -> Optional[Dict[int, int]]:
        if self.failed is not None:
            return None
        try:
            if self.handle is None:
                from sglang.srt.mem_ledger.flight_recorder import card_pin_unresolvable_without_cuda
                from sglang.srt.registry import nvml as registry_nvml

                if card_pin_unresolvable_without_cuda() is not None:
                    return None  # try again at a later mark, never create a context
                uuid = registry_nvml.current_device_uuid()
                import pynvml

                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            p = self.pynvml
            try:
                raw = p.nvmlDeviceGetComputeRunningProcesses_v3(self.handle)
            except AttributeError:
                raw = p.nvmlDeviceGetComputeRunningProcesses(self.handle)
            return {int(x.pid): int(x.usedGpuMemory or 0) for x in raw}
        except Exception as e:  # noqa: BLE001
            self.failed = f"{type(e).__name__}: {e}"
            logger.info("WEG2-VRAM-ACTUAL NVML unavailable (%s): nvml_pid stays null", self.failed)
            return None


def boot_key_of(rank_state_dir: Optional[str]) -> Optional[str]:
    """The part of a group's RankState directory every group of the SAME boot
    shares: ``<state>/rankstate/<G>`` -> ``<state>/rankstate``;
    ``<prefix>.<G>.log.rankstate`` -> ``<prefix>``."""
    if not rank_state_dir:
        return None
    d = str(rank_state_dir).rstrip("/")
    head, tail = os.path.split(d)
    if os.path.basename(head) == "rankstate" and len(tail) <= 2:
        return head
    if d.endswith(".log.rankstate"):
        stem = d[: -len(".log.rankstate")]
        base, _, _g = stem.rpartition(".")
        return base or stem
    return d


def _environ_of(pid: int) -> Dict[str, str]:
    with open(f"/proc/{int(pid)}/environ", "rb") as f:
        raw = f.read()
    out = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep:
            out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


class _Recorder:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.switch: Optional[bool] = None
        self.reset_state()

    def reset_state(self) -> None:
        self.nvml = _Nvml()
        self.block: Optional[VramBlock] = None
        self.attach = None  # RankState of the attach, None before it
        self.state_dir: Optional[str] = None
        self.seq = 0
        self.writes = 0
        self.ctx_baseline: Optional[int] = None
        self.peaks: Dict[str, Dict[str, int]] = {}
        self.marks: Dict[str, Dict[str, int]] = {}
        self.scheduler_ref: Optional[Callable[[], Any]] = None
        self.pid_class: Dict[int, str] = {}
        self.tag_list: Optional[Tuple[str, ...]] = None
        self.providers_cache: Optional[Dict[str, int]] = None
        #: id(model) -> its expert caches; ("params", id(model)) -> [bytes]
        #: (both walked once per model)
        self.expert_caches: Dict[Any, list] = {}
        #: hermetic tests: callables replacing the live readings
        self.fake_torch: Optional[Callable[[], Tuple[int, int]]] = None
        self.fake_nvml: Optional[Callable[[], Optional[Dict[int, int]]]] = None
        self.fake_tags: Optional[Callable[[], Tuple[Dict[str, int], FrozenSet[str]]]] = None
        self.fake_providers: Optional[Callable[[], Dict[str, int]]] = None
        self.pid: Optional[int] = None


_REC = _Recorder()


def reset() -> None:
    """Forget everything (tests; a re-exec'd process starts clean anyway)."""
    with _REC.lock:
        _REC.switch = None
        _REC.reset_state()


def write_count() -> int:
    return _REC.writes


def current_block() -> Optional[VramBlock]:
    return _REC.block


def _group() -> str:
    return (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper()


def _my_pid() -> int:
    return _REC.pid if _REC.pid is not None else os.getpid()


def _torch_now() -> Tuple[int, int]:
    if _REC.fake_torch is not None:
        return _REC.fake_torch()
    import torch

    if not torch.cuda.is_initialized():
        return 0, 0
    st = torch.cuda.memory_stats()
    return int(st.get("reserved_bytes.all.current", 0)), int(st.get("allocated_bytes.all.current", 0))


def _procs() -> Optional[Dict[int, int]]:
    if _REC.fake_nvml is not None:
        return _REC.fake_nvml()
    return _REC.nvml.procs()


def _tag_list() -> Tuple[str, ...]:
    if _REC.tag_list is None:
        tags = {"kv_cache", "cuda_graph", "weights", "weights_draft"}
        try:
            from sglang.srt.managers.weg2_memory_saver import weights_family_tags

            tags |= set(weights_family_tags())
        except Exception:  # noqa: BLE001
            pass
        _REC.tag_list = tuple(sorted(tags))
    return _REC.tag_list


def _tags_now() -> Tuple[Dict[str, int], FrozenSet[str]]:
    if _REC.fake_tags is not None:
        return _REC.fake_tags()
    from sglang.srt.utils.torch_memory_saver_adapter import _weg2_ring_symbol

    fn = _weg2_ring_symbol("tms_tag_bytes")
    tags: Dict[str, int] = {}
    if fn is not None:
        import ctypes

        fn.restype = ctypes.c_uint64
        fn.argtypes = [ctypes.c_char_p]
        for t in _tag_list():
            try:
                tags[t] = int(fn(t.encode()))
            except Exception:  # noqa: BLE001
                pass
    paused: FrozenSet[str] = frozenset()
    sched = _scheduler()
    wu = getattr(sched, "weight_updater", None) if sched is not None else None
    if wu is not None:
        paused = frozenset(str(t) for t in (getattr(wu, "offload_tags", None) or ()))
    return tags, paused


def _scheduler():
    ref = _REC.scheduler_ref
    return None if ref is None else ref()


def _expert_caches(model) -> list:
    """Every expert-offload cache of ``model``; the module walk runs once per
    model, later marks only read the cached list."""
    key = id(model)
    got = _REC.expert_caches.get(key)
    if got is None:
        got = []
        if hasattr(model, "modules"):
            for m in model.modules():
                cache = getattr(m, "_expert_offload", None)
                if isinstance(getattr(cache, "_resident", None), dict):
                    got.append(cache)
        _REC.expert_caches[key] = got
    return got


def expert_rows_bytes(cache) -> Tuple[int, int]:
    """``(resident, lru)`` bytes one cache's device buffers BACK now: R rows
    resident, the rest LRU/scratch/staging -- minus the H95c seat rows that
    are OFF (``seat_rows - seat_on``, unmapped by the D seat controller)."""
    bufs = getattr(cache, "_resident", None) or {}
    r_rows = int(getattr(cache, "resident_count", 0) or 0)
    seat_rows = int(getattr(cache, "seat_rows", 0) or 0)
    tables = getattr(cache, "_pool_tables", None)
    seat_on = int(getattr(tables, "seat_on", 0) or 0) if tables is not None else 0
    seat_off = max(0, seat_rows - seat_on)
    res_b = lru_b = 0
    for t in bufs.values():
        try:
            rows = int(t.shape[0])
            nbytes = int(t.numel()) * int(t.element_size())
        except Exception:  # noqa: BLE001
            continue
        if rows <= 0:
            continue
        per = nbytes // rows
        res = min(r_rows, rows)
        res_b += res * per
        lru_b += max(0, rows - res - seat_off) * per
    return res_b, lru_b


def _param_bytes(model) -> int:
    """Device bytes of ``model``'s parameters and buffers, once per model."""
    key = ("params", id(model))
    got = _REC.expert_caches.get(key)
    if got is None:
        n = 0
        seen = set()
        for t in list(model.parameters()) + list(model.buffers()):
            if not getattr(t, "is_cuda", False) or t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr())
            n += int(t.numel()) * int(t.element_size())
        got = [n]
        _REC.expert_caches[key] = got
    return int(got[0])


def _runner_providers(runner, *, draft: bool) -> Dict[str, int]:
    """Structural sizes of one model runner: expert rows, state pools, KV
    pool, parameter bytes (``draft_params`` for the draft runner)."""
    out = {"experts_resident": 0, "experts_lru": 0, "state_pools": 0, "kv_pool": 0,
           "target_params": 0, "draft_params": 0}
    model = getattr(runner, "model", None)
    if model is not None and hasattr(model, "parameters"):
        out["draft_params" if draft else "target_params"] = _param_bytes(model)
    if model is not None:
        for cache in _expert_caches(model):
            res_b, lru_b = expert_rows_bytes(cache)
            out["experts_resident"] += res_b
            out["experts_lru"] += lru_b
    GB = 1 << 30
    pool = getattr(runner, "token_to_kv_pool", None)
    mp = getattr(pool, "mamba_pool", None) or getattr(getattr(runner, "req_to_token_pool", None), "mamba_pool", None)
    mu = getattr(mp, "mem_usage", None)
    if mu:
        out["state_pools"] += int(float(mu) * GB)
    get_kv = getattr(pool, "get_kv_size_bytes", None)
    if callable(get_kv):
        try:
            kv = get_kv()
            out["kv_pool"] += int(sum(kv) if isinstance(kv, (tuple, list)) else kv)
        except Exception:  # noqa: BLE001
            pass
    return out


def _providers(refresh: bool) -> Dict[str, int]:
    if _REC.fake_providers is not None:
        return _REC.fake_providers()
    if not refresh and _REC.providers_cache is not None:
        return _REC.providers_cache
    out = {"experts_resident": 0, "experts_lru": 0, "state_pools": 0, "kv_pool": 0,
           "kv_arena_backed": 0, "target_params": 0, "draft_params": 0}
    sched = _scheduler()
    if sched is not None:
        dw = getattr(sched, "draft_worker", None)
        draft_runner = None
        if dw is not None:
            draft_runner = getattr(dw, "model_runner", None) or getattr(
                getattr(dw, "draft_worker", None), "model_runner", None)
        target_runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
        for runner, is_draft in ((target_runner, False), (draft_runner, True)):
            if runner is None or (is_draft and runner is target_runner):
                continue
            try:
                for k, v in _runner_providers(runner, draft=is_draft).items():
                    out[k] += v
            except Exception as e:  # noqa: BLE001
                logger.debug("WEG2-VRAM-ACTUAL provider skipped: %s", e)
    try:
        from sglang.srt.mem_ledger.flight_recorder import _kv_arena_view

        out["kv_arena_backed"] = int(_kv_arena_view().get("kv_arena_backed_bytes", 0) or 0)
    except Exception:  # noqa: BLE001
        pass
    _REC.providers_cache = out
    return out


def _classify_pid(pid: int) -> str:
    """``partner`` (another group of this boot) or ``foreign``; cached per pid."""
    c = _REC.pid_class.get(pid)
    if c is not None:
        return c
    mine = boot_key_of(os.environ.get("SGLANG_WEG2_RANK_STATE_DIR"))
    try:
        theirs = boot_key_of(_environ_of(pid).get("SGLANG_WEG2_RANK_STATE_DIR"))
    except OSError:
        theirs = None
    c = "partner" if (mine is not None and theirs == mine) else "foreign"
    _REC.pid_class[pid] = c
    return c


def _measure(name: str, refresh_providers: bool) -> Tuple[VramBlock, int]:
    """One mark: read, attribute, build the block. Returns (block, cost_us)."""
    t0 = time.perf_counter()
    reserved, allocated = _torch_now()
    procs = _procs()
    me = _my_pid()
    nvml_pid = None if procs is None else procs.get(me)
    tags, paused = _tags_now()
    prov = _providers(refresh_providers)
    if name == "pre_weight_load" and _REC.ctx_baseline is None and nvml_pid:
        _REC.ctx_baseline = max(0, int(nvml_pid) - int(reserved))
    reading = Reading(
        nvml_pid=nvml_pid, reserved=reserved, allocated=allocated, tags=tags, paused=paused,
        experts_resident=prov.get("experts_resident", 0), experts_lru=prov.get("experts_lru", 0),
        state_pools=prov.get("state_pools", 0), kv_pool=prov.get("kv_pool", 0),
        kv_arena_backed=prov.get("kv_arena_backed", 0), ctx_baseline=_REC.ctx_baseline,
        target_params=prov.get("target_params", 0), draft_params=prov.get("draft_params", 0),
    )
    actual, parts, sources, findings = attribute(reading)
    foreign = partner = None
    if procs is not None:
        foreign = partner = 0
        for pid, used in procs.items():
            if pid == me:
                continue
            if _classify_pid(pid) == "partner":
                partner += used
            else:
                foreign += used
    cost_us = int((time.perf_counter() - t0) * 1e6)
    block = VramBlock(
        actual=actual, nvml_pid_mib=_mib(nvml_pid), torch_reserved_mib=_mib(reserved),
        torch_allocated_mib=_mib(allocated), unattributed_parts=parts, findings=findings,
        foreign_mib=_mib(foreign), partner_mib=_mib(partner), sources=sources, mark=name,
        ts=time.time(),
    )
    return block, cost_us


def _write(block: VramBlock, name: str, cost_us: int) -> None:
    """Fold the per-mark row and the peaks in; write when an attach exists."""
    row = dict(_REC.marks.get(name) or {"n": 0})
    row.update(nvml_pid=block.nvml_pid_mib if block.nvml_pid_mib is not None else -1,
               reserved=block.torch_reserved_mib, allocated=block.torch_allocated_mib,
               cost_us=max(int(row.get("cost_us", 0)), int(cost_us)), n=int(row.get("n", 0)) + 1)
    row.setdefault("write_us", 0)
    _REC.marks[name] = row
    block = replace(block, marks={k: dict(v) for k, v in _REC.marks.items()},
                    peak_by_state={k: dict(v) for k, v in _REC.peaks.items()})
    _REC.block = block
    if _REC.attach is None or not _REC.state_dir:
        return
    _REC.seq += 1
    block = replace(block, seq=_REC.seq)
    _REC.block = block
    from sglang.srt.weg2 import rank_state as rs

    t0 = time.perf_counter()
    rs.write_rank_state(_replace_dataclass(_REC.attach, vram=block.to_dict()), _REC.state_dir)
    _REC.writes += 1
    row["write_us"] = max(int(row.get("write_us", 0)), int((time.perf_counter() - t0) * 1e6))


def on_mark(name: str, *, refresh_providers: bool = True) -> Optional[VramBlock]:
    """A named mark (boot phase, flip leg, attach). Never raises."""
    if not enabled():
        return None
    try:
        with _REC.lock:
            block, cost = _measure(name, refresh_providers)
            _write(block, name, cost)
            return _REC.block
    except Exception as e:  # noqa: BLE001 -- an instrument never breaks a boot
        logger.info("WEG2-VRAM-ACTUAL mark %s skipped (%s: %s)", name, type(e).__name__, e)
        return None


def attach(state, state_dir: Optional[str]):
    """The attach record (cache controller): remember it and hand it back
    carrying the current block. Off -> the record unchanged."""
    if not enabled() or not state_dir:
        return state
    with _REC.lock:
        _REC.attach = _replace_dataclass(state, vram=None)
        _REC.state_dir = state_dir
        try:
            block, cost = _measure("attach", True)
            row = dict(_REC.marks.get("attach") or {"n": 0})
            row.update(nvml_pid=block.nvml_pid_mib if block.nvml_pid_mib is not None else -1,
                       reserved=block.torch_reserved_mib, allocated=block.torch_allocated_mib,
                       cost_us=max(int(row.get("cost_us", 0)), cost), n=int(row.get("n", 0)) + 1,
                       write_us=int(row.get("write_us", 0)))
            _REC.marks["attach"] = row
            _REC.seq += 1
            block = replace(block, seq=_REC.seq, marks={k: dict(v) for k, v in _REC.marks.items()},
                            peak_by_state={k: dict(v) for k, v in _REC.peaks.items()})
            _REC.block = block
            _REC.writes += 1  # the caller writes this record
            return _replace_dataclass(state, vram=block.to_dict())
        except Exception as e:  # noqa: BLE001
            logger.info("WEG2-VRAM-ACTUAL attach block skipped (%s: %s)", type(e).__name__, e)
            return state


def bind_scheduler(scheduler) -> None:
    """run_scheduler_process, once: the scheduler whose weight updater knows
    the paused tags and whose runners hold the pools. Weakly held."""
    if not enabled():
        return
    import weakref

    try:
        _REC.scheduler_ref = weakref.ref(scheduler)
    except TypeError:
        _REC.scheduler_ref = lambda s=scheduler: s
    _REC.providers_cache = None


# -- WEG2-VRAM-PEAK windows --------------------------------------------------


def _window_key(phase: str, rows: int, leg: Optional[str], seats: int) -> Optional[str]:
    g = _group()
    if phase == "flip" and leg:
        return flip_state_key(g, leg)
    if phase == "idle":
        return None
    sched = _scheduler()
    if g == "P":
        stau = None
        if sched is not None:
            try:
                wq = getattr(sched, "waiting_queue", None) or ()
                rb = getattr(getattr(sched, "running_batch", None), "reqs", None) or ()
                stau = len(wq) + len(rb)
            except Exception:  # noqa: BLE001
                stau = None
        return p_state_key(rows if phase == "chunk" else 0, stau)
    stage = None
    if sched is not None:
        try:
            from sglang.srt.weg2.d_seat_vram import PHASE_ATTR

            st = getattr(sched, PHASE_ATTR, None)
            stage = None if st is None or getattr(st, "stage", None) is None else int(st.stage)
        except Exception:  # noqa: BLE001
            stage = None
    return d_state_key(seats, stage, rows if phase == "chunk" else 0)


def on_window(phase: str, *, peak_alloc: int, peak_reserved: int, start_alloc: Optional[int],
              rows: int = 0, leg: Optional[str] = None, seats: int = 0) -> bool:
    """The end of one WEG2-VRAM-PEAK window (bytes from its one memory_stats
    read; ``seats`` = the largest batch it saw). Folds the max per state key
    in memory; a full mark (NVML + write) only when a max changed -- also for
    a flip leg, whose RPC the front sends once per tag/wave. Returns True
    when it wrote. Never raises."""
    if not enabled():
        return False
    try:
        with _REC.lock:
            key = _window_key(phase, rows, leg, seats)
            if key is None:
                return False
            new = {"reserved_peak": _mib(peak_reserved), "allocated_peak": _mib(peak_alloc),
                   "transient": 0 if start_alloc is None else max(0, _mib(peak_alloc - start_alloc))}
            old = _REC.peaks.get(key)
            if old is None:
                row = dict(new, nvml_pid=-1, n=1)
                changed = True
            else:
                row = dict(old)
                row["n"] = int(old.get("n", 0)) + 1
                changed = False
                for k, v in new.items():
                    if v > int(old.get(k, 0)):
                        row[k] = v
                        changed = True
            _REC.peaks[key] = row
            if not changed:
                return False
            block, cost = _measure(f"window:{phase}" if phase != "flip" else key, phase == "flip")
            if block.nvml_pid_mib is not None:
                row["nvml_pid"] = max(int(row.get("nvml_pid", -1)), block.nvml_pid_mib)
            _write(block, "window" if phase != "flip" else key, cost)
            return True
    except Exception as e:  # noqa: BLE001
        logger.info("WEG2-VRAM-ACTUAL window skipped (%s: %s)", type(e).__name__, e)
        return False
