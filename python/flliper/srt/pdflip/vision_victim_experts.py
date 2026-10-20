# SPDX-License-Identifier: Apache-2.0
"""VISION-WEIGHTS AP3 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009 §1e/§2/§5/§6,
NF line): the victims of the transient tower are RESIDENT EXPERT ROWS of the
PP0 rank whose bytes already live in the expert store.

``--pdflip-vision-place weights`` on NF: the tower's parameters become views on
the device rows of the Platztausch EXTRA experts of P's PP0 layers. Those rows
are exactly the ones every wake loads from their fixed store slots
(``load_refill_rows`` over the layer's refill runs); the store is pinned and
already booked, so the borrow needs

* no D2H and no host image -- ``stash`` is a no-op, ``host_bytes`` is 0;
  instead ``verify_before_move`` compares the store's checksum with the
  device rows' before a byte moves (a stale store slot is W111b, not a
  W110c after the overwrite);
* the return = the wake's own refill: ``load_refill_rows`` with the SAME runs
  the rearm uses (``_rearm_targets``), clipped to the rows the tower touched.

Eligibility (plan §2, R1-R5), deterministic (the same set on every run):

* R1 only the TARGET model of this rank (``tp_worker.model_runner.model``) --
  the PP0 stage's own MoE layers; the draft (NEXTN, fully resident,
  ``FLLIPER_MOE_OFFLOAD_EXCLUDE_DRAFT``) is another model and never looked at.
* R2/R4 only rows with a STORE SLOT: a refill run ``(row0, slot0, n)`` with
  ``slot0 >= 0`` and ``slot0 + n`` inside the store tensor (its first dim is
  the slot count, ``FLLIPER_MOE_EXPERT_STORE_SLOT_FRACTION`` / the Karte's
  ``slots``). The Platztausch PREFIX (``common``, the rows the flip exchange
  moves) has no slot by construction of the Karte and is never a victim; a
  pad run (``slot0 < 0``) is not either. A layer marked
  ``_moe_offload_excluded`` is skipped.
* R3 ``join`` lands the deferred extra rows (``DeferredRowsFill.land_now``)
  and waits the expert copy streams (each cache's fetch stream, the pool /
  prefill-fetch-overlap side stream) -- before the move AND again before the
  return.
* R5 a candidate is a maximal run of consecutive rows of ONE expert attribute
  (w13 ~62.5 MiB per layer on the NF golden form); the core plans
  first-fit-decreasing over them with 256 B grain.

The core (``pdflip.vision_victim``) holds the order and the invariants: the
checksum before and after, W105b when the rows cannot hold the tower, W111b
when the plan does not match the live storages, W110c (fatal) when the return
fails or a checksum differs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

import msgspec
import torch

from flliper.srt.pdflip import vision_victim as vv
from flliper.srt.pdflip.vision_rank_stage import MIB, SLAB_ALIGN

logger = logging.getLogger(__name__)

KIND_EXPERTS = "experts"

#: (row0, slot0, n) -- one refill run of the Platztausch Karte
Run = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# 1. records and the pure run arithmetic
# ---------------------------------------------------------------------------


class RowRun(msgspec.Struct, frozen=True):
    """One victim candidate: rows ``[row0, row1)`` of ONE expert attribute
    of one layer, every row with a store slot (``runs`` = the layer's refill
    runs restricted to these rows)."""

    name: str
    layer: str
    attr: str
    row0: int
    row1: int
    row_bytes: int
    runs: Tuple[Run, ...]

    @property
    def nbytes(self) -> int:
        return (self.row1 - self.row0) * self.row_bytes


class StoreCensus(msgspec.Struct, frozen=True):
    """M0 numbers of the inventory: layers with store rows, candidates (row
    runs, one per layer and expert attribute), their bytes, and the rows
    refused (a pad run, a slot past the store's end, a buffer that cannot be
    refilled byte for byte), counted per attribute."""

    layers: int
    candidates: int
    nbytes: int
    refused_rows: int


def slot_runs(runs: Sequence[Run], *, store_slots: int) -> Tuple[Tuple[Run, ...], int]:
    """(runs whose every row has a store slot, rows refused). A pad run
    (``slot0 < 0``) has no slot; a run reaching past ``store_slots`` would
    read a row the store file does not hold (SLOT_FRACTION) -- both refused."""
    kept, refused = [], 0
    for z0, p0, n in runs:
        if int(p0) >= 0 and int(p0) + int(n) <= int(store_slots):
            kept.append((int(z0), int(p0), int(n)))
        else:
            refused += int(n)
    return tuple(sorted(kept)), refused


def row_ranges(runs: Sequence[Run]) -> List[Tuple[int, int]]:
    """Maximal ranges of consecutive rows covered by ``runs`` (sorted by row)."""
    out: List[Tuple[int, int]] = []
    for z0, _p0, n in sorted(runs):
        if out and out[-1][1] == z0:
            out[-1] = (out[-1][0], z0 + n)
        else:
            out.append((z0, z0 + n))
    return out


def clip_runs(runs: Sequence[Run], *, row0: int, row1: int) -> Tuple[Run, ...]:
    """``runs`` restricted to rows ``[row0, row1)``, the slots shifted along."""
    out = []
    for z0, p0, n in runs:
        lo, hi = max(z0, row0), min(z0 + n, row1)
        if lo < hi:
            out.append((lo, p0 + (lo - z0), hi - lo))
    return tuple(out)


def rows_touched(*, row0: int, row_bytes: int, offset: int, nbytes: int) -> Tuple[int, int]:
    """The rows a segment ``[offset, offset + nbytes)`` of a run overlaps."""
    first = row0 + offset // row_bytes
    last = row0 + (offset + nbytes + row_bytes - 1) // row_bytes
    return first, last


# ---------------------------------------------------------------------------
# 2. the live layer targets (the rearm's own branch)
# ---------------------------------------------------------------------------


class LayerTarget(msgspec.Struct, frozen=True):
    """One offload layer as the rearm sees it: its name, its cache (None
    before the lazy install -- then the presplit buffers serve), its refill
    runs and ``entries`` = ((attr, device buffer, store tensor), ...)."""

    layer: str
    cache: Any
    runs: Tuple[Run, ...]
    entries: Tuple[Tuple[str, Any, Any], ...]


def model_targets(model: torch.nn.Module) -> List[LayerTarget]:
    """Every offload layer of ``model`` with refill runs, in module order,
    through the SAME branch as the wake's rearm and the H31 prefetch
    (``expert_offload._rearm_targets``) -- the victims are the rows those
    load. A layer excluded from the offload (the fully resident draft block)
    has no refill runs and yields nothing."""
    from flliper.srt.layers.moe.expert_offload import _rearm_targets

    out = []
    for name, module in model.named_modules():
        target = _rearm_targets(module)
        if target is None:
            continue
        cache, runs, entries = target
        if runs and entries:
            out.append(LayerTarget(layer=name, cache=cache, runs=tuple(runs),
                                   entries=tuple((a, b, s) for a, b, s in entries)))
    return out


def _row_bytes(rows: torch.Tensor) -> torch.Tensor:
    """The bytes of a block of rows as a flat uint8 tensor (a view when the
    block is contiguous, which a slice of leading rows of a contiguous
    buffer is)."""
    return rows.contiguous().reshape(-1).view(torch.uint8)


def _run_base(buf: torch.Tensor, row0: int, row_bytes: int) -> int:
    """Byte offset of row ``row0`` of ``buf`` inside its storage."""
    return int(buf.storage_offset()) * int(buf.element_size()) + int(row0) * int(row_bytes)


def _eligible_entry(buf: Any, spill: Any) -> bool:
    """A buffer whose rows are contiguous bytes and a store of the same row
    form -- else its rows cannot be refilled byte for byte."""
    return (isinstance(buf, torch.Tensor) and isinstance(spill, torch.Tensor)
            and buf.dim() >= 1 and buf.is_contiguous() and spill.dim() == buf.dim()
            and tuple(spill.shape[1:]) == tuple(buf.shape[1:]) and spill.dtype == buf.dtype)


def layer_row_runs(target: LayerTarget, ordinal: int) -> Tuple[List[RowRun], int]:
    """(candidates, rows refused) of one layer: per attribute the maximal row
    ranges of its slot runs."""
    out, refused = [], 0
    for attr, buf, spill in target.entries:
        if not _eligible_entry(buf, spill):
            refused += sum(n for _z, _p, n in target.runs)
            continue
        kept, bad = slot_runs(target.runs, store_slots=int(spill.shape[0]))
        refused += bad
        row_bytes = int(buf[0].numel()) * int(buf.element_size())
        for r0, r1 in row_ranges(kept):
            out.append(RowRun(name=f"{ordinal:03d}:{target.layer}.{attr}[{r0}:{r1}]", layer=target.layer,
                              attr=attr, row0=r0, row1=r1, row_bytes=row_bytes,
                              runs=clip_runs(kept, row0=r0, row1=r1)))
    return out, refused


# ---------------------------------------------------------------------------
# 3. the source
# ---------------------------------------------------------------------------


class ExpertRowVictims:
    """``VictimSource`` over the resident expert rows with a store slot.
    ``targets`` yields the live layers (default: :func:`model_targets` of the
    model), ``refill`` is ``load_refill_rows``, ``joins`` the R3 waits -- all
    injectable for the desk."""

    kind = KIND_EXPERTS

    def __init__(self, *, targets: Callable[[], List[LayerTarget]],
                 refill: Callable[..., int], joins: Sequence[Callable[[], Any]]):
        self._targets = targets
        self._refill = refill
        self._joins = tuple(joins)
        self._borrowed: List[Tuple[RowRun, Tuple[str, Any, Any], vv.VictimSegment]] = []

    # -- inventory ----------------------------------------------------------
    def _live(self) -> Tuple[Dict[str, Tuple[RowRun, Tuple[str, Any, Any]]], StoreCensus]:
        live: Dict[str, Tuple[RowRun, Tuple[str, Any, Any]]] = {}
        seen = set()
        refused = nbytes = layers = 0
        for i, tgt in enumerate(self._targets()):
            cands, bad = layer_row_runs(tgt, i)
            refused += bad
            layers += 1 if cands else 0
            entry_of = {attr: (attr, buf, spill) for attr, buf, spill in tgt.entries}
            for c in cands:
                entry = entry_of[c.attr]
                key = (int(entry[1].untyped_storage().data_ptr()), _run_base(entry[1], c.row0, c.row_bytes))
                if key in seen:  # R2: an alias of rows already counted
                    continue
                seen.add(key)
                live[c.name] = (c, entry)
                nbytes += c.nbytes
        return live, StoreCensus(layers=layers, candidates=len(live), nbytes=nbytes,
                                 refused_rows=refused)

    @staticmethod
    def _candidate(run: RowRun, buf: torch.Tensor) -> vv.VictimCandidate:
        key = int(buf.untyped_storage().data_ptr()) + _run_base(buf, run.row0, run.row_bytes)
        off = (-key) % SLAB_ALIGN
        return vv.VictimCandidate(name=run.name, key=key, offset=off, nbytes=run.nbytes - off,
                                  storage_nbytes=run.nbytes)

    def census(self) -> StoreCensus:
        return self._live()[1]

    def inventory(self) -> List[vv.VictimCandidate]:
        live, _census = self._live()
        out = []
        for name in sorted(live):
            run, (_a, buf, _s) = live[name]
            cand = self._candidate(run, buf)
            if cand.nbytes > 0:
                out.append(cand)
        return out

    def views(self, segments: Sequence[vv.VictimSegment]) -> List[torch.Tensor]:
        """The live uint8 view of each planned segment; W111b when a segment
        no longer names the same rows at the same address."""
        live, _census = self._live()
        out, borrowed = [], []
        for seg in segments:
            hit = live.get(seg.name)
            cand = None if hit is None else self._candidate(hit[0], hit[1][1])
            if cand is None or cand.key != seg.key or cand.storage_nbytes != seg.storage_nbytes:
                raise vv.VisionVictimPlanRefused(
                    f"{vv.W_VICTIM_PLAN_REFUSED}: expert rows {seg.name} at 0x{seg.key:x} "
                    f"({seg.storage_nbytes} B) are not live rows of that size any more")
            run, entry = hit
            base = _run_base(entry[1], run.row0, run.row_bytes)
            whole = vv.storage_view(entry[1])[base:base + run.nbytes]
            out.append(whole[seg.offset:seg.offset + seg.nbytes])
            borrowed.append((run, entry, seg))
        self._borrowed = borrowed
        return out

    # -- the legs -----------------------------------------------------------
    @property
    def host_bytes(self) -> int:
        return 0

    def join(self) -> None:
        for wait in self._joins:
            wait()

    def verify_before_move(self, views: Sequence[torch.Tensor]) -> None:
        """Review S2: what comes back is the STORE's rows, what the checksum
        guards is the DEVICE's rows. Before a byte moves, the checksum of
        every store piece the return would load must equal the checksum of
        the device rows it replaces; a stale or rewritten store slot is W111b
        here (nothing moved, the rig intact) instead of a W110c crash-stop
        after the rows were already overwritten. Only the rows the tower
        touches are compared (the same rows ``restore`` reloads)."""
        for run, (attr, buf, spill), seg in self._borrowed:
            r0, r1 = rows_touched(row0=run.row0, row_bytes=run.row_bytes, offset=seg.offset,
                                  nbytes=seg.nbytes)
            for z0, p0, n in clip_runs(run.runs, row0=r0, row1=r1):
                dev = vv.segment_checksum(_row_bytes(buf[z0:z0 + n]))
                host = vv.segment_checksum(_row_bytes(spill[p0:p0 + n]))
                if dev != host:
                    raise vv.VisionVictimPlanRefused(
                        f"{vv.W_VICTIM_PLAN_REFUSED}: store slots {p0}..{p0 + n - 1} of {run.layer}.{attr} "
                        f"differ from the device rows {z0}..{z0 + n - 1} they would be loaded back into; "
                        "nothing moved")

    def stash(self, views: Sequence[torch.Tensor]) -> None:
        """No-op: the bytes already live in the pinned store (plan R4)."""
        return None

    def restore(self, views: Sequence[torch.Tensor]) -> None:
        """The rows the tower touched back from their store slots -- the
        rearm's ``load_refill_rows`` over the clipped runs, on the current
        stream (the lease's checksum reads after it on the same stream)."""
        self.join()
        for run, entry, seg in self._borrowed:
            r0, r1 = rows_touched(row0=run.row0, row_bytes=run.row_bytes, offset=seg.offset,
                                  nbytes=seg.nbytes)
            self._refill([entry], clip_runs(run.runs, row0=r0, row1=r1), layer_id=run.layer)

    def release(self) -> None:
        self._borrowed = []

    def shape_tower(self, module, tower, largest_run) -> vv.SplitMap:
        """No row split on NF: one w13 run of a layer (~62.5 MiB on the
        golden form) holds the largest tower tensor (40.5 MiB); a shorter
        run is a named W105b of the core's plan."""
        return {}


# ---------------------------------------------------------------------------
# 4. the R3 joins and the registration
# ---------------------------------------------------------------------------


def _join_deferred_rows() -> int:
    from flliper.srt.layers.moe.expert_offload import deferred_rows_fill

    return deferred_rows_fill().land_now(why="before the vision stage borrows expert rows (VISION-WEIGHTS)")


def _join_copy_streams(targets: Callable[[], List[LayerTarget]]) -> None:
    """Each cache's fetch stream and the one pool / prefill-fetch-overlap
    stream (only if it exists): they write scratch rows, never a victim row,
    but R3 names no exception."""
    from flliper.srt.layers.moe import expert_offload as eo

    for tgt in targets():
        if tgt.cache is not None and tgt.cache._stream is not None:
            tgt.cache._stream.synchronize()
    if eo._POOL_PREFETCH_STREAM is not None:
        eo._POOL_PREFETCH_STREAM.synchronize()


def _target_model(scheduler) -> torch.nn.Module:
    return scheduler.tp_worker.model_runner.model


def _applies(scheduler) -> bool:
    return bool(model_targets(_target_model(scheduler)))


def _build(scheduler) -> ExpertRowVictims:
    from flliper.srt.layers.moe.expert_offload import load_refill_rows

    model = _target_model(scheduler)

    def targets() -> List[LayerTarget]:
        return model_targets(model)

    source = ExpertRowVictims(targets=targets, refill=load_refill_rows,
                              joins=(_join_deferred_rows, lambda: _join_copy_streams(targets)))
    c = source.census()
    logger.info("%s victim=%s store_layers=%d store_runs=%d store_mib=%.1f refused_rows=%d "
                "(resident rows with a store slot of this rank's layers; the Platztausch prefix "
                "has no slot and is never a victim)",
                vv.W_VICTIM_STORE, KIND_EXPERTS, c.layers, c.candidates, c.nbytes / MIB, c.refused_rows)
    return source


vv.register_source(KIND_EXPERTS, _applies, _build)
