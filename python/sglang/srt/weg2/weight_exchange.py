# SPDX-License-Identifier: Apache-2.0
"""Weg-2 #1273 -- the weight-byte EXCHANGE.

Spec of record: ``/spinning/gpu-arb/weg2/WEG2_REUSE_SPEC_0908.md``.

Under ``--weg2-weight-source exchange`` the weights region is opened with
``enable_cpu_backup=False``, which makes ``pause`` a pure unmap and ``resume``
a pure remap (core.cpp:229-231, :362-364).  No host granule is acquired on the
weights path, so the host ring is not created at all; between the destination's
remap and its first use the exchange injects the bytes VRAM->VRAM.

THIS FILE IS BUILT BY TWO SLICES.  They are kept in separate, labelled
sections so the two branches meet without arguing about line order:

* **S1 -- the plan** (``build_plan`` and the ``XchgDesc`` tiling, spec 6/S1).
  NOT in this file yet.  S2 consumes exactly one thing from it and nothing
  else: ``{tag: {parameter name, ...}}`` (see :data:`PlanNames`).
* **S2 -- coverage arming (W51) and the draft tag** (spec 6/S2, 4.1), below.

Everything here is hermetic: no CUDA, no torch_memory_saver, no checkpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import torch

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.managers.weg2_memory_saver import (
    Weg2XchgCoverageRefused,
    is_weights_family_tag,
    layer_id_from_module_name,
    weight_chunk_tag,
)

logger = logging.getLogger(__name__)

MIB = 1024 * 1024


# ===========================================================================
# THE MODE.  Boot-scoped and region-scoped (spec section 1.1): one boolean,
# decided once, at model_runner.py's region().
# ===========================================================================

WEIGHT_SOURCE_RING = "ring"
WEIGHT_SOURCE_EXCHANGE = "exchange"

#: MINIMAL INTERFACE, TODO(S6, spec section 6/S6): the user-facing flag is
#: ``--weg2-weight-source {ring|exchange}`` on ``server_args``, and the
#: launcher publishes it to BOTH groups' worker processes.  Until S6 lands that
#: flag, the mode is read from this env var -- the same channel the launcher
#: already uses for the chunk geometry (``SGLANG_WEG2_WEIGHT_CHUNK_LAYERS``,
#: weg2_memory_saver.py), so no new transport is invented here.  S6 replaces
#: the body of :func:`weight_source` with the server-arg read and adds the flag
#: to ``p_form_key``'s blacklist BY NAME (spec risk R8 / the #1275 admin-key
#: trap): a new flag not named there gives every boot its own form and W48 then
#: discards the source boot's weight statement every boot.
WEIGHT_SOURCE_ENV = "SGLANG_WEG2_WEIGHT_SOURCE"


def weight_source() -> str:
    """``ring`` (today, byte for byte) or ``exchange``.  Default ``ring``."""
    value = (os.environ.get(WEIGHT_SOURCE_ENV, "") or "").strip().lower()
    return (
        WEIGHT_SOURCE_EXCHANGE
        if value == WEIGHT_SOURCE_EXCHANGE
        else WEIGHT_SOURCE_RING
    )


def exchange_armed() -> bool:
    return weight_source() == WEIGHT_SOURCE_EXCHANGE


@contextmanager
def weight_source_for_test(source: str) -> Iterator[str]:
    """Set the mode for the duration of a block.  Tests only."""
    previous = os.environ.get(WEIGHT_SOURCE_ENV)
    os.environ[WEIGHT_SOURCE_ENV] = source
    try:
        yield source
    finally:
        if previous is None:
            os.environ.pop(WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[WEIGHT_SOURCE_ENV] = previous


# ===========================================================================
# S2 -- THE DRAFT TAG (spec section 4.1)
# ===========================================================================


def weights_region_tag_for(*, is_draft_model_runner: bool) -> str:
    """The tag ``model_runner``'s weights region is opened with.

    Group D's base tag carries the ENTIRE NEXTN draft runner -- measured
    1382/1311/1311 MiB on boot weg2sb4 (``Load weight end.
    type=Qwen3_5ForCausalLMMTP ... mem usage=1.35/1.28/1.28 GB``), of which the
    checkpoint's ``mtp.*`` term is only 0.396 GiB; the rest is embed/head
    shards the draft runner re-materialises.  Group P carries no
    ``--speculative-*`` in this form, so NONE of those bytes has a VRAM source
    on the other side.  Under ``exchange`` they get their own tag, which
    ``is_weights_family_tag`` does not match, so they are never in a leg, never
    in a census and never in a wave, and every REMAINING destination byte has a
    source -- which is what lets the ring go to zero rather than to MTP-only.

    Cost: 1.35/1.28/1.28 GiB permanently resident per card (spec section 5).

    TWO DEVIATIONS FROM THE SPEC'S PSEUDOCODE, both deliberate:

    1. **Gated on the mode.**  Spec 4.1 writes the tag switch unconditionally;
       spec 1.1 requires ``ring`` to stay "today, byte for byte".  An
       unconditional switch changes the ring arm too -- the drafter would stop
       being paused, moving ~1.3 GiB/card from the host ring into permanent
       residency on a path nobody asked to change.  So the switch is armed by
       :func:`exchange_armed`.
    2. **``is_draft_model_runner``, not ``is_draft_worker``.**  The spec names
       the construction gate; the tree names it a trap in its own words
       (model_runner.py:514-521): ``is_draft_worker`` has THREE producers -- a
       speculative draft worker, the #274 dual-group lane, and the #631
       phase-flip TP stack -- and "only the first actually holds draft
       weights"; asking the construction gate when you mean draft-NESS "has now
       cost five separate boot failures".  A phase-flip TP stack tagged
       ``weights_draft`` would take the FULL TARGET MODEL out of the weights
       family.
    """
    if is_draft_model_runner and exchange_armed():
        return GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    return GPU_MEMORY_TYPE_WEIGHTS


RESIDENT_LINE_PREFIX = "WEG2-XCHG-RESIDENT"


def resident_line(*, rank: int, tag: str, mib: float) -> str:
    """The acceptance line for a tag that is RESIDENT across the flip.

    Per rank, because a rank knows its own card and no other: the spec's
    ``mib=1382/1311/1311`` is the operator's three-card read of three of these
    lines, not a line any single rank can emit truthfully.
    """
    return (
        f"{RESIDENT_LINE_PREFIX} tag={tag} rank={int(rank)} mib={float(mib):.1f} "
        f"in_family={'yes' if is_weights_family_tag(tag) else 'no'}"
    )


# ===========================================================================
# S2 -- COVERAGE ARMING (W51, spec section 2.4 rule 4 and section 6/S2)
# ===========================================================================

COVER_LINE_PREFIX = "WEG2-XCHG-COVER"
COVERAGE_REFUSAL_MARKER = "W51 Weg2XchgCoverageRefused"

#: MINIMAL INTERFACE, TODO(S1, branch weg2/xchg-s1-0908): the plan half of this
#: module must expose its parameter population in exactly this shape --
#: ``{tag: {parameter name, ...}}`` over ``named_parameters()`` names, one
#: entry per exchanged tag.  S2 consumes this and nothing else of the plan, so
#: the two slices meet on a mapping rather than on a class.
PlanNames = Mapping[str, AbstractSet[str]]

#: What ``tms_tag_bytes`` answers for one tag, in bytes; 0 means THE SAVER
#: COULD NOT ANSWER and is printed as such, never read as "this tag is empty"
#: (weight_updater.py:355-376).
TagBytesFn = Callable[[str], int]

PARAMETER = "parameter"
BUFFER = "buffer"
ATTRIBUTE = "attribute"


@dataclass(frozen=True)
class LiveTensor:
    """One live tensor found under an exchanged tag, and how it was found."""

    name: str
    module_path: str
    kind: str  # PARAMETER | BUFFER | ATTRIBUTE
    tag: str
    nbytes: int
    storage_key: Tuple[Any, ...]
    dtype: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class TagCoverage:
    """The arithmetic for one exchanged tag, on one rank."""

    rank: int
    tag: str
    planned_bytes: int
    buffers_bytes: int
    tms_bytes: int
    uncovered: Tuple[LiveTensor, ...]
    n_parameters: int
    n_buffers: int
    n_attributes: int

    @property
    def slack_bytes(self) -> int:
        """``tms_tag_bytes`` minus what this walk accounts for.

        PRINTED, NEVER COMPARED FOR EQUALITY (spec section 6/S2).  Allocator
        overhang is real and measured at +0.08 to +0.58 GiB per rank, so an
        equality assert would refuse every boot.  It can also be NEGATIVE --
        that is the saver answering 0 because the running hook has no
        ``tms_tag_bytes`` symbol, an ABSENCE the reader must see rather than a
        deficit the code invents a rule about.
        """
        return int(self.tms_bytes) - int(self.planned_bytes) - int(self.buffers_bytes)

    def cover_line(self) -> str:
        """The acceptance line, spec field order verbatim.

        Fields after ``uncovered=`` are APPENDED, never interleaved: the
        greppable prefix stays exactly what the spec names, and the counts that
        follow are the DENOMINATOR of ``uncovered`` -- a population number
        without its population is the trap this campaign keeps paying for.
        """
        return (
            f"{COVER_LINE_PREFIX} rank={int(self.rank)} tag={self.tag} "
            f"planned_mib={self.planned_bytes / MIB:.1f} "
            f"buffers_mib={self.buffers_bytes / MIB:.1f} "
            f"tms_mib={self.tms_bytes / MIB:.1f} "
            f"slack_mib={self.slack_bytes / MIB:.1f} "
            f"uncovered={len(self.uncovered)} "
            f"params={self.n_parameters} buffers={self.n_buffers} "
            f"attrs={self.n_attributes} "
            f"tms_answered={'yes' if self.tms_bytes else 'no'}"
        )


def tag_of_parameter_name(name: str) -> str:
    """The torch_memory_saver tag a tensor of this NAME was allocated under.

    Name-derived and allocation-derived are the same thing here, which is the
    only reason a per-tag census can be compared against a per-name plan: every
    layer is constructed inside ``weight_chunk_scope(idx)``
    (utils/common.py:2017) and every post-load repack inside
    ``weight_chunk_scope(layer_id_from_module_name(name))``
    (model_loader/loader.py:941), both of which take the chunk from the layer
    id.  The clamp for a layer beyond the last chunk is ``weight_chunk_tag``'s,
    not a second copy of it.
    """
    layer_id = layer_id_from_module_name(name)
    if layer_id is None:
        return GPU_MEMORY_TYPE_WEIGHTS
    return weight_chunk_tag(layer_id) or GPU_MEMORY_TYPE_WEIGHTS


def _storage_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identity of the ALLOCATION behind a tensor, so a view is not a second
    tensor.  ``self.lm_head = self.model.embed_tokens`` (qwen3_5_mtp.py:288) is
    the shape in the tree; charging its bytes twice would inflate the census
    comparison and, worse, a stray view of covered bytes would be reported as
    an uncovered page that does not exist."""
    try:
        storage = tensor.untyped_storage()
        nbytes = int(storage.nbytes())
        if nbytes > 0:
            return ("storage", int(storage.data_ptr()), nbytes)
    except Exception:  # noqa: BLE001 -- meta/fake tensors have no storage
        pass
    return ("object", id(tensor))


def _nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return int(tensor.numel()) * int(tensor.element_size())


def _join(module_path: str, attr: str) -> str:
    return f"{module_path}.{attr}" if module_path else attr


def walk_live_tensors(model: torch.nn.Module) -> List[LiveTensor]:
    """Every live tensor of the model, with the tag it lives under.

    Three populations, and the third is the reason this function exists:

    * ``named_parameters()`` -- what the plan transports;
    * ``named_buffers()`` -- what ``_export_static_state`` carries across the
      flip (weight_updater.py:1291 -> :1755); the largest is the rope
      ``cos_sin_cache`` (rotary_embedding/base.py:173, measured +300/+181/+210
      MiB on D's ``weights_0``);
    * every module's ``__dict__`` -- a plain ``self.x = torch.zeros(...)``
      is neither a Parameter nor a buffer, appears in NO standard iterator, and
      is exactly the byte the exchange would leave undefined.

    BOUND OF THE CHECK, stated rather than assumed: a tensor reachable only
    through a container attribute (a list, a dict, a dataclass field on a
    module) is not found by this walk.  It refuses what it can see and names
    its population on the log line; it does not claim to be exhaustive over
    every reachable object graph.
    """
    out: List[LiveTensor] = []
    for name, param in model.named_parameters():
        out.append(
            LiveTensor(
                name=name,
                module_path=name.rsplit(".", 1)[0] if "." in name else "",
                kind=PARAMETER,
                tag=tag_of_parameter_name(name),
                nbytes=_nbytes(param),
                storage_key=_storage_key(param),
                dtype=str(param.dtype),
                shape=tuple(param.shape),
            )
        )
    for name, buf in model.named_buffers():
        if buf is None:
            continue
        out.append(
            LiveTensor(
                name=name,
                module_path=name.rsplit(".", 1)[0] if "." in name else "",
                kind=BUFFER,
                tag=tag_of_parameter_name(name),
                nbytes=_nbytes(buf),
                storage_key=_storage_key(buf),
                dtype=str(buf.dtype),
                shape=tuple(buf.shape),
            )
        )
    for module_path, module in model.named_modules():
        for attr, value in list(vars(module).items()):
            if not isinstance(value, torch.Tensor):
                continue
            name = _join(module_path, attr)
            out.append(
                LiveTensor(
                    name=name,
                    module_path=module_path,
                    kind=ATTRIBUTE,
                    tag=tag_of_parameter_name(name),
                    nbytes=_nbytes(value),
                    storage_key=_storage_key(value),
                    dtype=str(value.dtype),
                    shape=tuple(value.shape),
                )
            )
    return out


def build_coverage(
    model: torch.nn.Module,
    *,
    rank: int,
    planned_names_by_tag: PlanNames,
    tag_bytes: TagBytesFn,
) -> Dict[str, TagCoverage]:
    """The per-tag arithmetic, over the exchanged tags only.

    An exchanged tag is a WEIGHTS-FAMILY tag: the base tag plus
    ``weights_<integer>``.  ``weights_draft`` is deliberately not one, so the
    drafter's bytes are never charged to a family tag's slack and never
    counted as an uncovered page.
    """
    live = walk_live_tensors(model)

    tags = {t.tag for t in live if is_weights_family_tag(t.tag)}
    tags |= {t for t in planned_names_by_tag if is_weights_family_tag(t)}

    rows: Dict[str, TagCoverage] = {}
    for tag in sorted(tags):
        planned_names = set(planned_names_by_tag.get(tag, ()))
        of_tag = [t for t in live if t.tag == tag]

        covered_storage = set()
        planned_bytes = 0
        buffers_bytes = 0
        uncovered: List[LiveTensor] = []
        n_par = n_buf = n_attr = 0

        # Parameters and buffers first: they define WHICH allocations are
        # accounted for, and only then can an attribute be judged as an alias
        # of one of them rather than as a page of its own.
        for t in of_tag:
            if t.kind == PARAMETER:
                n_par += 1
                if t.name in planned_names:
                    if t.storage_key not in covered_storage:
                        covered_storage.add(t.storage_key)
                        planned_bytes += t.nbytes
                else:
                    uncovered.append(t)
            elif t.kind == BUFFER:
                n_buf += 1
                if t.storage_key not in covered_storage:
                    covered_storage.add(t.storage_key)
                    buffers_bytes += t.nbytes
        for t in of_tag:
            if t.kind != ATTRIBUTE:
                continue
            n_attr += 1
            if t.storage_key not in covered_storage:
                uncovered.append(t)

        rows[tag] = TagCoverage(
            rank=int(rank),
            tag=tag,
            planned_bytes=planned_bytes,
            buffers_bytes=buffers_bytes,
            tms_bytes=int(tag_bytes(tag) or 0),
            uncovered=tuple(uncovered),
            n_parameters=n_par,
            n_buffers=n_buf,
            n_attributes=n_attr,
        )
    return rows


def coverage_refusal_message(rows: Mapping[str, TagCoverage]) -> str:
    """The W51 text: every uncovered tensor named, with its module path."""
    parts: List[str] = []
    for tag in sorted(rows):
        for t in rows[tag].uncovered:
            parts.append(
                f"tag={tag} kind={t.kind} name={t.name} "
                f"module={t.module_path or '<root>'} "
                f"dtype={t.dtype} shape={list(t.shape)} mib={t.nbytes / MIB:.3f}"
            )
    return (
        f"{COVERAGE_REFUSAL_MARKER}: {len(parts)} live tensor(s) under an exchanged "
        "tag are neither a plan Parameter nor a registered buffer, so the "
        "exchange has no source for their pages and the destination would serve "
        "whatever its arena held. " + " | ".join(parts) + " -- either the plan is "
        "extended to cover them, or that tag runs "
        "--weg2-weight-source ring (spec section 6/S2 stop-loss). The exchange "
        "does not ship a tag it cannot fully account for."
    )


def arm_coverage(
    model: torch.nn.Module,
    *,
    rank: int,
    planned_names_by_tag: PlanNames,
    tag_bytes: TagBytesFn,
    log: Optional[Callable[[str], None]] = None,
    draft_tag_mib: Optional[float] = None,
) -> Dict[str, TagCoverage]:
    """Arm the exchange for this rank, or refuse by name.

    CALL SITE, TODO(S6): at the end of weight loading, and again in the wake
    RPC's preamble BEFORE the first ``resume`` -- which is where an abandon
    still costs nothing because neither side's VRAM has been mutated (spec
    section 3.6).  It is not called from ``model_runner`` on this branch: the
    plan is S1's and the mode flag is S6's, and a call site that can only raise
    because its inputs do not exist yet is worse than no call site.

    Returns the rows so the caller can log or re-check them; raises
    :class:`Weg2XchgCoverageRefused` when any tag has an uncovered tensor.
    """
    emit = log if log is not None else logger.info
    rows = build_coverage(
        model,
        rank=rank,
        planned_names_by_tag=planned_names_by_tag,
        tag_bytes=tag_bytes,
    )
    for tag in sorted(rows):
        emit(rows[tag].cover_line())
    if draft_tag_mib is not None:
        emit(
            resident_line(
                rank=rank, tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT, mib=draft_tag_mib
            )
        )
    if any(rows[tag].uncovered for tag in rows):
        raise Weg2XchgCoverageRefused(coverage_refusal_message(rows))
    return rows
