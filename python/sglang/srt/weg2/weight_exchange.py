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
  else: ``{tag: {parameter name: planned bytes}}`` (see :data:`PlanBytes`),
  where the bytes are the sum of that parameter's ``XchgDesc.nbytes``.
* **S2 -- coverage arming (W51) and the draft tag** (spec 6/S2, 4.1), below.

Everything here is hermetic: no CUDA, no torch_memory_saver, no checkpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
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

RUNNER_SHAPE_REFUSAL_MARKER = "W60 Weg2XchgRunnerShapeUnknown"


class Weg2XchgRunnerShapeUnknown(RuntimeError):
    """W60 -- a secondary ModelRunner this predicate cannot classify.

    ``is_draft_worker`` is a CONSTRUCTION gate with several producers and only
    one of them holds draft weights (model_runner.py:514-521).  The tag switch
    of spec section 4.1 must therefore classify the runner, not read the gate;
    a shape it cannot classify is refused by name rather than guessed, because
    both guesses are wrong in a way that shows up only at the next flip:
    guessing DRAFT takes a full target model out of the weights family (never
    paused, never exchanged, permanently resident, no refusal names it), and
    guessing PRIMARY puts unsourced bytes into it (W58).

    W60 is free per the spec section 7 census (highest assigned W51, this
    branch) and per ``test_weg2_wcode_uniqueness_1263``.
    """


#: The four shapes a ModelRunner can have, as far as WHICH WEIGHTS IT HOLDS is
#: concerned.  This is the question ``is_draft_worker`` is repeatedly mistaken
#: for; naming the shapes is what makes the mistake un-writable.
SHAPE_PRIMARY = "primary"
SHAPE_DRAFT = "draft"
SHAPE_PHASE_FLIP_TP_STACK = "phase_flip_tp_stack"
SHAPE_DUAL_GROUP_LANE = "dual_group_lane"


@dataclass(frozen=True)
class RunnerShape:
    """The ModelRunner construction flags that decide which weights it holds.

    Read off the runner rather than passed as one boolean, because the boolean
    was the defect: ``is_draft_worker and not is_phase_flip_tp_stack`` subtracts
    ONE of the wrong producers and leaves the #274 dual-group lane -- built with
    ``is_draft_worker=True, is_phase_flip_tp_stack=False``
    (model_runner.py:486-489) and holding the assembled FULL-WIDTH TARGET hull
    (``build_lane_model``, :2496-2513) -- classified as a drafter.
    """

    is_draft_worker: bool
    is_phase_flip_tp_stack: bool = False
    is_dual_group_lane: bool = False
    speculative_configured: bool = False

    @classmethod
    def of(cls, runner: Any) -> RunnerShape:
        server_args = getattr(runner, "server_args", None)
        return cls(
            is_draft_worker=bool(getattr(runner, "is_draft_worker", False)),
            is_phase_flip_tp_stack=bool(
                getattr(runner, "is_phase_flip_tp_stack", False)
            ),
            is_dual_group_lane=bool(getattr(runner, "is_dual_group_lane", False)),
            speculative_configured=bool(
                getattr(server_args, "speculative_algorithm", None)
            ),
        )

    def as_fields(self) -> str:
        return (
            f"is_draft_worker={self.is_draft_worker} "
            f"is_phase_flip_tp_stack={self.is_phase_flip_tp_stack} "
            f"is_dual_group_lane={self.is_dual_group_lane} "
            f"speculative_configured={self.speculative_configured}"
        )


def classify_runner(shape: RunnerShape) -> Optional[str]:
    """Which weights this runner holds, or ``None`` when that is not derivable.

    The producers of ``is_draft_worker``, each subtracted BY NAME:

    * ``is_phase_flip_tp_stack`` -- #631's secondary TP decode stack, the full
      TARGET model rebuilt under the flip group's geometry;
    * ``is_dual_group_lane`` -- #274's lane, the assembled full-width TARGET
      hull (and, with ``is_dual_group_lane_draft``, its NEXTN head, whose vocab
      shells ALIAS the lane target's -- one allocation may not straddle two
      tags, so the whole lane stays in the family);
    * a speculative draft worker -- the only producer that holds draft weights,
      and it exists only when this boot carries a speculative config.

    ``None`` is a fourth producer this function has never heard of: it sets the
    construction gate, sets none of the known exclusions, and the boot carries
    no speculative config, so there is no shape to attribute it to.
    """
    if not shape.is_draft_worker:
        return SHAPE_PRIMARY
    if shape.is_phase_flip_tp_stack:
        return SHAPE_PHASE_FLIP_TP_STACK
    if shape.is_dual_group_lane:
        return SHAPE_DUAL_GROUP_LANE
    if shape.speculative_configured:
        return SHAPE_DRAFT
    return None


def runner_shape_refusal_message(shape: RunnerShape) -> str:
    return (
        f"{RUNNER_SHAPE_REFUSAL_MARKER}: this ModelRunner is constructed with "
        f"is_draft_worker=True but is neither the #631 phase-flip TP stack, nor "
        f"the #274 dual-group lane, nor a speculative draft worker (this boot "
        f"carries no speculative config): {shape.as_fields()}. Under "
        f"--weg2-weight-source exchange the weights region tag decides whether "
        f"these bytes are exchanged at all, and neither answer is safe for a "
        f"shape nobody classified -- the draft tag would take a full target "
        f"model out of the weights family, the base tag would put bytes with no "
        f"VRAM source into it (W58). Classify the new producer in "
        f"classify_runner() or run this boot with --weg2-weight-source ring."
    )


def weights_region_tag_for(shape: RunnerShape) -> str:
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
    2. **A classified shape, not ``is_draft_worker``.**  The spec names the
       construction gate; the tree names that gate a trap in its own words
       (model_runner.py:514-521), and subtracting only the phase-flip stack from
       it -- the first form of this function -- reproduced the trap one producer
       later for the #274 lane.  See :func:`classify_runner`.
    """
    kind = classify_runner(shape)
    if kind is None:
        if exchange_armed():
            raise Weg2XchgRunnerShapeUnknown(runner_shape_refusal_message(shape))
        # Under ``ring`` nothing changes for anybody, including a shape this
        # predicate cannot name: today's path, byte for byte.
        return GPU_MEMORY_TYPE_WEIGHTS
    if kind == SHAPE_DRAFT and exchange_armed():
        return GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    return GPU_MEMORY_TYPE_WEIGHTS


RESIDENT_LINE_PREFIX = "WEG2-XCHG-RESIDENT"


def resident_line(*, tag: str, mib: float, rank: int, mode: str) -> str:
    """The acceptance line for the tag this runner's weights actually live under.

    An OBSERVATION, not a restatement of a constant (refuter F1): the first form
    of this line hardcoded ``tag=weights_draft`` and then computed
    ``in_family`` from that literal, so ``in_family=no`` was a property of a
    string this module owns -- it could never print ``yes``, and it printed the
    same under ``ring``, where the drafter IS in the family.  The tag is now
    the one the region was opened with, ``mib`` is the saver's own per-tag
    census for THAT tag, and the mode is on the line, so the line distinguishes
    an armed exchange from an unarmed one.

    Per rank, because a rank knows its own card and no other: the spec's
    ``mib=1382/1311/1311`` is the operator's three-card read of three of these
    lines, not a line any single rank can emit truthfully.  Field order puts
    ``tag= mib= in_family=`` adjacent, which is the adjacency the spec greps.
    """
    return (
        f"{RESIDENT_LINE_PREFIX} tag={tag} mib={float(mib):.1f} "
        f"in_family={'yes' if is_weights_family_tag(tag) else 'no'} "
        f"rank={int(rank)} mode={mode}"
    )


def roll_forward_weights_tag(*, has_draft_shard: bool) -> Optional[str]:
    """The tag W57's roll-forward reload may open its ONE region with.

    ``None`` means REFUSE.  ``_weg2_wake_reload_weights``
    (weight_updater.py:531-706) refills BOTH shards of a process through ONE
    ``update_weights_from_disk`` inside ONE region -- the code says so at
    :576-597 and refuses any arrangement where the two shards disagree.  Under
    ``exchange`` the two shards want two different region tags (the drafter's
    is out of the family by construction), and one region carries one tag, so
    there is no tag that is right for both: the base tag re-tags the drafter's
    repacked parameters back into the weights family and the next flip then has
    a destination range with no VRAM source (W58), while the draft tag does the
    same thing to the main shard in the other direction.

    Refusing is the only answer this slice can give.  Making the roll-forward
    work under ``exchange`` means one region per runner, which is a change to
    the reload's shape and belongs to S6 with W57 (spec section 3.6).
    """
    if not exchange_armed():
        return GPU_MEMORY_TYPE_WEIGHTS
    if has_draft_shard:
        return None
    return GPU_MEMORY_TYPE_WEIGHTS


def roll_forward_refusal_message() -> str:
    return (
        "--weg2-weight-source exchange gives the NEXTN/MTP draft shard its own "
        f"region tag ({GPU_MEMORY_TYPE_WEIGHTS_DRAFT}, spec section 4.1), and "
        "the backup-OFF wake refills BOTH shards of this process through ONE "
        "update_weights_from_disk inside ONE region "
        "(weight_updater.py:576-597). One region carries one tag, so this "
        "roll-forward would re-tag the drafter's repacked parameters into the "
        "weights family and the next flip would have a destination range with "
        "no VRAM source (W58). Refusing rather than re-tagging. OWNER: S6 -- "
        "W57's roll-forward needs one region per runner (spec section 3.6)."
    )


# ===========================================================================
# S2 -- COVERAGE ARMING (W51, spec section 2.4 rule 4 and section 6/S2)
# ===========================================================================

COVER_LINE_PREFIX = "WEG2-XCHG-COVER"
COVERAGE_REFUSAL_MARKER = "W51 Weg2XchgCoverageRefused"

#: MINIMAL INTERFACE, TODO(S1, branch weg2/xchg-s1-0908): the plan half of this
#: module must expose its parameter population in exactly this shape --
#: ``{tag: {parameter name: PLANNED BYTES}}``, one entry per exchanged tag,
#: over ``named_parameters()`` names, where the bytes are the sum of that
#: parameter's ``XchgDesc.nbytes`` (the descriptors the exchange will actually
#: move), NOT the live tensor's storage size.
#:
#: NAMES ALONE ARE NOT ENOUGH, and that is the whole point (refuter F3): with a
#: name set, a parameter counts as covered the moment its NAME appears, and the
#: bytes charged are the live tensor's -- so a plan whose ``in_proj_qkvz``
#: carries three device sub-blocks instead of four (spec section 2.2 / risk R5,
#: the case that writes ``k`` over ``q`` on ranks 1 and 2) passes with
#: ``uncovered=0`` and a plausible slack.  The arming check for SILENT
#: wrongness must be able to see a partially covered parameter, so the plan
#: hands over its byte claim and this check compares it against the storage.
PlanBytes = Mapping[str, Mapping[str, int]]

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
class ShortParameter:
    """A parameter the plan carries, but not all of."""

    name: str
    tag: str
    planned_bytes: int
    live_bytes: int


@dataclass(frozen=True)
class TagCoverage:
    """The arithmetic for one exchanged tag, on one rank."""

    rank: int
    tag: str
    mode: str
    planned_bytes: int
    buffers_bytes: int
    tms_bytes: int
    uncovered: Tuple[LiveTensor, ...]
    short: Tuple[ShortParameter, ...]
    missing: Tuple[str, ...]
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

    @property
    def ok(self) -> bool:
        return not (self.uncovered or self.short or self.missing)

    def cover_line(self) -> str:
        """The acceptance line, spec field order verbatim.

        Fields after ``uncovered=`` are APPENDED, never interleaved: the
        greppable prefix stays exactly what the spec names, and the counts that
        follow are the DENOMINATOR of ``uncovered`` -- a population number
        without its population is the trap this campaign keeps paying for.
        ``planned_mib`` is the PLAN's byte claim, so a plan that covers a
        parameter only partly shows up here as slack rather than vanishing.
        """
        return (
            f"{COVER_LINE_PREFIX} rank={int(self.rank)} tag={self.tag} "
            f"planned_mib={self.planned_bytes / MIB:.1f} "
            f"buffers_mib={self.buffers_bytes / MIB:.1f} "
            f"tms_mib={self.tms_bytes / MIB:.1f} "
            f"slack_mib={self.slack_bytes / MIB:.1f} "
            f"uncovered={len(self.uncovered)} "
            f"short={len(self.short)} missing={len(self.missing)} "
            f"params={self.n_parameters} buffers={self.n_buffers} "
            f"attrs={self.n_attributes} "
            f"tms_answered={'yes' if self.tms_bytes else 'no'} "
            f"mode={self.mode}"
        )


def tag_of_parameter_name(
    name: str, *, region_tag: str = GPU_MEMORY_TYPE_WEIGHTS
) -> str:
    """The torch_memory_saver tag a tensor of this NAME was allocated under.

    THE REGION WINS OVER THE NAME, and it has to (review F1 / refuter F4).
    Inside the BASE weights region, name-derived and allocation-derived are the
    same thing: every layer is constructed inside ``weight_chunk_scope(idx)``
    (utils/common.py:2017) and every post-load repack inside
    ``weight_chunk_scope(layer_id_from_module_name(name))``
    (model_loader/loader.py:941), both of which take the chunk from the layer
    id, and that identity is the only reason a per-tag census can be compared
    against a per-name plan.  Inside ANY OTHER weights region the identity
    breaks in exactly one direction: ``weight_chunk_scope`` is a no-op there
    (weg2_memory_saver.py, #1273 S2), so the drafter's one-layer
    ``Qwen3_5ForCausalLM`` block (qwen3_5_mtp.py:277-283) is ALLOCATED under
    ``weights_draft`` while its name still says ``layers.0.``.  A region-blind
    reading returns ``weights_0`` for those bytes and would then census a
    family tag that does not exist in that process.

    The clamp for a layer beyond the last chunk is ``weight_chunk_tag``'s, not
    a second copy of it.
    """
    if region_tag != GPU_MEMORY_TYPE_WEIGHTS:
        return region_tag
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


def walk_live_tensors(
    model: torch.nn.Module, *, region_tag: str = GPU_MEMORY_TYPE_WEIGHTS
) -> List[LiveTensor]:
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

    def _add(name: str, tensor: torch.Tensor, kind: str, module_path: str) -> None:
        out.append(
            LiveTensor(
                name=name,
                module_path=module_path,
                kind=kind,
                tag=tag_of_parameter_name(name, region_tag=region_tag),
                nbytes=_nbytes(tensor),
                storage_key=_storage_key(tensor),
                dtype=str(tensor.dtype),
                shape=tuple(tensor.shape),
            )
        )

    for name, param in model.named_parameters():
        _add(name, param, PARAMETER, name.rsplit(".", 1)[0] if "." in name else "")
    for name, buf in model.named_buffers():
        if buf is None:
            continue
        _add(name, buf, BUFFER, name.rsplit(".", 1)[0] if "." in name else "")
    for module_path, module in model.named_modules():
        for attr, value in list(vars(module).items()):
            if not isinstance(value, torch.Tensor):
                continue
            _add(_join(module_path, attr), value, ATTRIBUTE, module_path)
    return out


def build_coverage(
    model: torch.nn.Module,
    *,
    rank: int,
    planned_bytes_by_tag: PlanBytes,
    tag_bytes: TagBytesFn,
    region_tag: str = GPU_MEMORY_TYPE_WEIGHTS,
    mode: Optional[str] = None,
) -> Dict[str, TagCoverage]:
    """The per-tag arithmetic, over the exchanged tags only.

    An exchanged tag is a WEIGHTS-FAMILY tag: the base tag plus
    ``weights_<integer>``.  ``weights_draft`` is deliberately not one, so the
    drafter's bytes are never charged to a family tag's slack and never
    counted as an uncovered page -- and under a draft region there is no family
    tag at all, so this returns no rows rather than censusing tags that do not
    exist in that process.

    Three refusal populations, not one:

    * ``uncovered`` -- a live tensor under an exchanged tag that is neither a
      plan parameter nor a registered buffer: a page the destination never
      receives;
    * ``short`` -- a plan parameter whose byte claim does not equal the live
      storage: the partially-tiled parameter, which a name-only check cannot
      see (risk R5);
    * ``missing`` -- a plan parameter with no live tensor on this rank: a plan
      built against a different shard geometry.
    """
    mode = mode if mode is not None else weight_source()
    live = walk_live_tensors(model, region_tag=region_tag)

    tags = {t.tag for t in live if is_weights_family_tag(t.tag)}
    tags |= {t for t in planned_bytes_by_tag if is_weights_family_tag(t)}

    rows: Dict[str, TagCoverage] = {}
    for tag in sorted(tags):
        planned_of_tag = dict(planned_bytes_by_tag.get(tag, {}))
        of_tag = [t for t in live if t.tag == tag]

        covered_storage = set()
        seen_names = set()
        planned_bytes = 0
        buffers_bytes = 0
        uncovered: List[LiveTensor] = []
        short: List[ShortParameter] = []
        n_par = n_buf = n_attr = 0

        # Parameters and buffers first: they define WHICH allocations are
        # accounted for, and only then can an attribute be judged as an alias
        # of one of them rather than as a page of its own.
        for t in of_tag:
            if t.kind == PARAMETER:
                n_par += 1
                if t.name in planned_of_tag:
                    seen_names.add(t.name)
                    claim = int(planned_of_tag[t.name])
                    if claim != t.nbytes:
                        short.append(
                            ShortParameter(
                                name=t.name,
                                tag=tag,
                                planned_bytes=claim,
                                live_bytes=t.nbytes,
                            )
                        )
                    if t.storage_key not in covered_storage:
                        covered_storage.add(t.storage_key)
                        # The PLAN's bytes, never the model's: a plan that
                        # tiles only three of four device sub-blocks must move
                        # this number, not the storage size it failed to cover.
                        planned_bytes += claim
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
            mode=mode,
            planned_bytes=planned_bytes,
            buffers_bytes=buffers_bytes,
            tms_bytes=int(tag_bytes(tag) or 0),
            uncovered=tuple(uncovered),
            short=tuple(short),
            missing=tuple(sorted(set(planned_of_tag) - seen_names)),
            n_parameters=n_par,
            n_buffers=n_buf,
            n_attributes=n_attr,
        )
    return rows


def coverage_refusal_message(rows: Mapping[str, TagCoverage]) -> str:
    """The W51 text: every refused tensor named, with its module path."""
    parts: List[str] = []
    for tag in sorted(rows):
        row = rows[tag]
        for t in row.uncovered:
            parts.append(
                f"UNCOVERED tag={tag} kind={t.kind} name={t.name} "
                f"module={t.module_path or '<root>'} "
                f"dtype={t.dtype} shape={list(t.shape)} mib={t.nbytes / MIB:.3f}"
            )
        for s in row.short:
            parts.append(
                f"SHORT tag={tag} name={s.name} "
                f"planned_mib={s.planned_bytes / MIB:.3f} "
                f"live_mib={s.live_bytes / MIB:.3f} "
                f"delta_mib={(s.live_bytes - s.planned_bytes) / MIB:.3f}"
            )
        for name in row.missing:
            parts.append(f"MISSING tag={tag} name={name}")
    return (
        f"{COVERAGE_REFUSAL_MARKER}: {len(parts)} finding(s). A live tensor "
        "under an exchanged tag that is neither a plan Parameter nor a "
        "registered buffer (UNCOVERED), a plan Parameter whose byte claim does "
        "not equal its live storage (SHORT), or a plan Parameter with no live "
        "tensor on this rank (MISSING) all mean the same thing: the exchange "
        "has no source for some destination page and the destination would "
        "serve whatever its arena held. " + " | ".join(parts) + " -- either the "
        "plan is extended to cover them, or that tag runs "
        "--weg2-weight-source ring (spec section 6/S2 stop-loss). The exchange "
        "does not ship a tag it cannot fully account for."
    )


@dataclass(frozen=True)
class CoverageVote:
    """This rank's arming verdict.  A VOTE, not a decision.

    #802 / "Raenge nie uneins" (refuter F5): the first form of this check
    raised ``Weg2XchgCoverageRefused`` straight out of the arming call.  At the
    call site spec section 6/S2 names -- the end of weight loading -- there is
    no group fence in scope, so one rank whose plan is short would die at boot
    while the other five continued into a collective that no longer has six
    members.  The derivation stays rank-local; the DECISION belongs to whoever
    holds a fence, so this type crosses that boundary and
    :func:`refuse_if_not_ok` is called only where a fence exists (the wake
    RPC's preamble, spec section 3.6, S6).
    """

    rank: int
    mode: str
    region_tag: str
    rows: Dict[str, TagCoverage]
    ok: bool
    reason: str


def refuse_if_not_ok(vote: CoverageVote) -> CoverageVote:
    """Raise W51 for a failing vote.  ONLY where a group fence covers it."""
    if not vote.ok:
        raise Weg2XchgCoverageRefused(vote.reason)
    return vote


def arm_coverage(
    model: torch.nn.Module,
    *,
    rank: int,
    planned_bytes_by_tag: PlanBytes,
    tag_bytes: TagBytesFn,
    log: Optional[Callable[[str], None]] = None,
    region_tag: str = GPU_MEMORY_TYPE_WEIGHTS,
) -> CoverageVote:
    """Compute this rank's arming vote and emit the acceptance lines.

    NEVER RAISES for a coverage finding -- see :class:`CoverageVote`.  Callers
    that hold a fence pass the result to :func:`refuse_if_not_ok`.
    """
    emit = log if log is not None else logger.info
    mode = weight_source()
    rows = build_coverage(
        model,
        rank=rank,
        planned_bytes_by_tag=planned_bytes_by_tag,
        tag_bytes=tag_bytes,
        region_tag=region_tag,
        mode=mode,
    )
    for tag in sorted(rows):
        emit(rows[tag].cover_line())
    ok = all(rows[tag].ok for tag in rows)
    return CoverageVote(
        rank=int(rank),
        mode=mode,
        region_tag=region_tag,
        rows=rows,
        ok=ok,
        reason="" if ok else coverage_refusal_message(rows),
    )


# ---------------------------------------------------------------------------
# THE WIRING.  An instrument nobody calls measures nothing.
# ---------------------------------------------------------------------------

#: The plan half (S1) registers its byte population here and S6's flag arms the
#: mode; this seam is what lets the check be WIRED before either exists.  A
#: SEAM, not a second bookkeeping: nothing is stored here but the callable that
#: answers "what does the plan carry for this model", and the answer is
#: recomputed at every call.
_PLAN_PROVIDER: Optional[Callable[[Any], PlanBytes]] = None

#: This rank's standing boot vote, for the fenced re-check to fold into the
#: group's decision (spec section 3.6: the RPC preamble re-checks coverage
#: BEFORE the first ``resume``, where an abandon still costs nothing).
_BOOT_VOTE: Optional[CoverageVote] = None


def register_plan_provider(fn: Optional[Callable[[Any], PlanBytes]]) -> None:
    """Register (or, with ``None``, clear) the plan's byte population source."""
    global _PLAN_PROVIDER
    _PLAN_PROVIDER = fn


def plan_provider() -> Optional[Callable[[Any], PlanBytes]]:
    return _PLAN_PROVIDER


def boot_vote() -> Optional[CoverageVote]:
    """This rank's arming vote from the end of weight loading, or ``None``."""
    return _BOOT_VOTE


def _record_boot_vote(vote: Optional[CoverageVote]) -> None:
    global _BOOT_VOTE
    _BOOT_VOTE = vote


NO_PLAN_REASON = (
    f"{COVERAGE_REFUSAL_MARKER}: --weg2-weight-source exchange is armed but no "
    "plan provider is registered on this rank, so NOTHING is accounted for: "
    "every weight page of every exchanged tag would be filled from a plan that "
    "does not exist. S1 registers the plan (register_plan_provider) and S6 "
    "arms the mode; a boot that has the mode without the plan is refused here "
    "rather than at the first flip."
)


def arm_coverage_at_load(
    model: Optional[torch.nn.Module],
    *,
    rank: int,
    tag_bytes: TagBytesFn,
    region_tag: str,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[CoverageVote]:
    """THE CALL SITE, spec section 6/S2: the end of weight loading.

    A no-op under ``ring`` -- no lines, no walk, no cost on the default path.
    Under ``exchange`` it emits this runner's ``WEG2-XCHG-RESIDENT`` line, and
    for a runner whose weights are IN the exchanged family it also builds the
    coverage rows, emits one ``WEG2-XCHG-COVER`` line per tag, and records the
    vote.

    It does NOT raise: there is no group fence at the end of weight loading
    (refuter F5).  The refusal text is logged with its marker so an operator
    sees it at boot, and the vote is recorded for the fenced re-check in the
    wake RPC's preamble, which is where §3.6 says an abandon still costs
    nothing.
    """
    emit = log if log is not None else logger.info
    if not exchange_armed():
        _record_boot_vote(None)
        return None
    mode = weight_source()

    def _bytes(tag: str) -> int:
        try:
            return int(tag_bytes(tag) or 0)
        except Exception:  # noqa: BLE001 -- an absent instrument is an absence
            return 0

    emit(
        resident_line(
            tag=region_tag, mib=_bytes(region_tag) / MIB, rank=rank, mode=mode
        )
    )
    if region_tag != GPU_MEMORY_TYPE_WEIGHTS:
        # This runner's weights are out of the exchanged family by construction
        # (spec section 4.1).  There is nothing for the exchange to cover here,
        # and nothing for it to get wrong; the RESIDENT line above is the whole
        # statement.
        vote = CoverageVote(
            rank=int(rank),
            mode=mode,
            region_tag=region_tag,
            rows={},
            ok=True,
            reason="",
        )
        _record_boot_vote(vote)
        return vote

    provider = plan_provider()
    if provider is None or model is None:
        vote = CoverageVote(
            rank=int(rank),
            mode=mode,
            region_tag=region_tag,
            rows={},
            ok=False,
            reason=NO_PLAN_REASON,
        )
        _record_boot_vote(vote)
        logger.error("%s", vote.reason)
        return vote

    vote = arm_coverage(
        model,
        rank=rank,
        planned_bytes_by_tag=provider(model),
        tag_bytes=_bytes,
        log=emit,
        region_tag=region_tag,
    )
    _record_boot_vote(vote)
    if not vote.ok:
        logger.error("%s", vote.reason)
    return vote
