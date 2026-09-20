"""Switch between TWO P layouts inside the RUNNING PP processes.

User order 2026-09-20: "wir muessen im prozess switchen koennen, da wir nie
wissen wie viel noch prefillt werden muss (es koennen ja waehrend des laufenden
prefills noch weitere pending dazukommen). aber ja, zwei layouts duerften dafuer
reichen. nur auf das ganz schnelle P layout switchen wenn der tok/s gewinn so
gross ist ueber die noch laufenden pending token, dass es den laengeren flip
spaeter rechtfertigt."

WHAT THIS IS NOT, AND WHY THE DISTINCTION IS THE WHOLE DESIGN
------------------------------------------------------------
Two mechanisms already exist in the tree and NEITHER answers the order:

* :mod:`sglang.srt.model_executor.layout_boundary` (#704 slice 1a-ii) moves the
  executed range at runtime and copies NOTHING -- "load wide, run narrow": every
  rank holds the UNION of both rungs resident.  That is exactly what the order
  cannot use.  The motive here is to FREE VRAM on stage 0 (the 5090) so D's
  bytes and D's KV stay resident across the prefill and the later P->D flip is
  shorter.  A union keeps stage 0 at ``max(counts0)`` layers under both rungs,
  so it frees nothing and the flip is not shortened at all.  This module is the
  byte-MOVING half that #704 deliberately left out; it reuses #704's actuator,
  its observers and its rollback rather than restating them.
* :mod:`sglang.srt.planner.boot_layout` records the runtime per-family flip as
  CLOSED (VERDICT_485_REMAINDER): a layout change priced as a whole-arena
  host->device refill costs ~1575 ms, constant in distance, and needs ~17,500
  rounds to repay.  That verdict prices a DIFFERENT move and its own premise
  says so -- "whole-arena", "host->device", "constant in distance".  The move
  here is BOUNDARY-LOCAL (only the layers that change owner) and card->card
  over BAR1, so its cost is proportional to the boundary distance and never
  touches host DRAM.  The gate is not evaded, it does not cover this case; a
  band-local move that priced out like a whole-arena refill would be refused
  by the same arithmetic, and :func:`decide` is where that arithmetic lives.

THE THREE THINGS THE ORDER ASKS FOR
-----------------------------------
1. The stage boundary as a RUNTIME variable.  It already is one:
   ``model._start_layer`` / ``_end_layer`` back the properties the decoder
   forward re-reads every pass (``models/qwen3_5.py:1669-1674``, ``:1720``), and
   ``make_layers`` builds a FULL-LENGTH ModuleList with ``PPMissingLayer``
   placeholders outside the owned range (``utils/common.py:2086-2094``), so
   layer indices are GLOBAL on every rank and a boundary change is not an index
   shift.  #704's ``LayoutBoundaryActuator`` is the actuation; this module
   supplies the bytes it needs to be allowed to move.
2. The bytes, per layer, card to card.  Priced from the rank manifests
   (:func:`layer_bytes_from_manifests`), which record what THIS rank's loader
   actually materialised -- weights, scales, biases, norms alike -- rather than
   from a per-layer hand number.  Moved with the same primitive the flip legs
   use, ``DeviceOps.memcpy_async`` (``weight_exchange_transport.py:655``),
   through the same deposit/collect shape as ``weight_exchange_bounce.py:917``
   and ``:941``, over a BAR1 peer window instead of a host buffer.
3. The rule.  :func:`decide`.

THE BAND IS THE UNIT OF FREED VRAM, NOT THE LAYER
-------------------------------------------------
The one fact that decides which second layout is worth having.  Weight
allocations are tagged per CHUNK BAND, not per layer:
``weight_chunk_tag(layer_id) = layer_id // layers_per_chunk``
(``managers/weg2_memory_saver.py:2316-2321``), and the memory saver pauses and
resumes whole TAGS.  So a stage gives VRAM back only for a band it stops owning
ENTIRELY; a band it still holds one layer of stays resident in full.  This is
the same geometry mismatch that killed boot weg2dk4 -- ``chunk_tag_cards``
(``weg2_memory_saver.py:2345``) exists because the tag index hid it -- and it
means a second layout whose boundary is NOT band-aligned can move bytes and
free nothing.  :func:`band_aligned_candidates` enumerates the ones that can,
and :func:`vram_freed_bytes` is what a candidate is judged on.

EVERY NUMBER HERE IS INJECTED, NONE IS WRITTEN DOWN
---------------------------------------------------
Layer bytes come from the manifests, stage times from
``planner.pp_cut.PrefillTiming`` (ONE object for both layouts, so the two sides
of every comparison share a measurement basis by construction -- the mixed-unit
class ``launcher.derive_x_star`` refuses by name), link rates from the measured
BAR1 table, flip seconds from the boot's own ``flip_total=`` lines.  A missing
input is a refusal, never a default: :class:`UnpricedLink`,
:class:`MissingLayerBytes`, :class:`UnpricedFlip`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "ACTIONS",
    "ACTION_STAY",
    "ACTION_TO_FAST",
    "ACTION_TO_LEAN",
    "ACTION_FAST_THEN_LEAN",
    "BandGeometry",
    "LayerBytes",
    "LayerMove",
    "LinkRates",
    "MissingLayerBytes",
    "MovePlan",
    "MoveOp",
    "PLayout",
    "PLayoutRankDisagree",
    "PLayoutSwitchError",
    "Plan",
    "RingState",
    "SwitchCalibration",
    "SwitchNotQuiescent",
    "SwitchVerdict",
    "UnalignedLayout",
    "UnpricedFlip",
    "UnpricedLink",
    "accept_pp0_verdict",
    "band_aligned_candidates",
    "bands_freed",
    "course_breakeven_tokens",
    "breakeven_tokens",
    "decide",
    "emit_move_ops",
    "layer_bytes_from_manifests",
    "layer_index_of",
    "move_seconds",
    "plan_layer_moves",
    "pp0_broadcast",
    "preconditions_digest",
    "prefill_seconds",
    "require_switchable",
    "run_move_ops",
    "run_move_ops_guarded",
    "require_breakable_route",
    "require_union_fits",
    "placement_switch_seconds",
    "slowest_hop_seconds",
    "union_kv_costs",
    "union_layout",
    "vram_freed_bytes",
    "MoverNotCapturable",
    "RANK_INDEXED_VECTORS",
    "StagePermutation",
    "StagePermutationIncomplete",
    "UnionKvCost",
    "UnionKvTooCostly",
]


# ---------------------------------------------------------------------------
# Refusals.  Each names the missing measurement, because the alternative is a
# confident number with no basis -- the class the INDIKATOR law is about.
# ---------------------------------------------------------------------------


class PLayoutSwitchError(RuntimeError):
    """Base: a P-layout switch that cannot be priced or cannot be honoured."""


class UnpricedLink(PLayoutSwitchError):
    """W120 -- a move was planned across a link whose rate was never measured."""


class MissingLayerBytes(PLayoutSwitchError):
    """W121 -- a layer that changes owner has no byte count in the inventory."""


class UnpricedFlip(PLayoutSwitchError):
    """W122 -- a layout was offered with no measured flip time of its own."""


class UnalignedLayout(PLayoutSwitchError):
    """W123 -- a layout whose boundaries free no whole band on the stage that
    the switch exists to relieve."""


class SwitchNotQuiescent(PLayoutSwitchError):
    """W124 -- a switch asked for while the PP ring still carries work."""


class PLayoutRankDisagree(PLayoutSwitchError):
    """W125 -- the ranks of group P do not agree on the switch decision.

    The law (memory ``raenge-nie-uneins``, and #968: PP0 is authoritative,
    downstream is verdict-FREE): a state-changing decision is taken ONCE for the
    whole group.  A switch that happens on two stages of three does not leave a
    slower pipeline, it leaves layers owned twice and layers owned by nobody --
    ``validate_world_tiling`` (``layout_boundary.py:84``) names both -- and the
    second of those produces plausible output from a shallower model rather than
    an error.  So a detected disagreement is a STOP on EVERY rank, never a
    compensation and never a retry on the odd one out.
    """


# ---------------------------------------------------------------------------
# The layout itself.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PLayout:
    """One contiguous PP stage map: how many layers each stage owns.

    CONTIGUOUS ON PURPOSE.  The gapped form exists (``SGLANG_PP_LAYER_SET``,
    ``distributed/utils.py:2157``) and is REFUSED for this use by boot weg2gp1
    (2026-09-08): 3 of 6 determined-answer probes diverge from the contiguous
    control in both graph modes against an A/A floor of 0, and the gapped map
    also lost on both axes it was chosen for (world pool -45.2 %, +89.8 % per
    chunk on the binding stage).  A boundary move needs no gaps anyway -- that
    is what makes it cheap -- so this type does not offer the form that gate
    refuses.
    """

    name: str
    counts: Tuple[int, ...]

    def __post_init__(self) -> None:
        counts = tuple(int(c) for c in self.counts)
        object.__setattr__(self, "counts", counts)
        if not self.name or not str(self.name).strip():
            raise UnalignedLayout(
                "a P layout needs a name; it is what the ranks agree on."
            )
        if len(counts) < 1:
            raise UnalignedLayout(f"layout {self.name!r} has no stages.")
        if any(c < 1 for c in counts):
            raise UnalignedLayout(
                f"layout {self.name!r} counts {counts}: every stage needs at "
                "least one layer. A stage with zero layers is not a cheaper "
                "pipeline, it is a stage whose KV pool divides by zero "
                "(pp_cut.py rev 4, the HybridLinearKVPool full_layer_nums path)."
            )

    # -- derived geometry ---------------------------------------------------

    @property
    def n_stages(self) -> int:
        return len(self.counts)

    @property
    def n_layers(self) -> int:
        return int(sum(self.counts))

    @property
    def bounds(self) -> Tuple[int, ...]:
        """Cumulative one-past-last layer of each stage."""
        out: List[int] = []
        acc = 0
        for c in self.counts:
            acc += int(c)
            out.append(acc)
        return tuple(out)

    def range_of(self, stage: int) -> Tuple[int, int]:
        """``(start_layer, end_layer)`` of ``stage`` -- #704's rung range."""
        stage = int(stage)
        if not 0 <= stage < self.n_stages:
            raise UnalignedLayout(
                f"layout {self.name!r} has {self.n_stages} stages; asked for "
                f"stage {stage}."
            )
        bounds = self.bounds
        start = bounds[stage - 1] if stage else 0
        return start, bounds[stage]

    def ranges(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self.range_of(s) for s in range(self.n_stages))

    def owner_of(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.n_layers:
            raise UnalignedLayout(
                f"layer {layer_id} is outside layout {self.name!r} "
                f"([0,{self.n_layers}))."
            )
        for stage, bound in enumerate(self.bounds):
            if layer_id < bound:
                return stage
        raise AssertionError("unreachable: bounds[-1] == n_layers")

    def layers_of(self, stage: int) -> Tuple[int, ...]:
        start, end = self.range_of(stage)
        return tuple(range(start, end))

    def as_ratio(self) -> str:
        """The ``--pp-layer-ratio`` string this layout is."""
        return ",".join(str(c) for c in self.counts)


def _require_same_shape(a: PLayout, b: PLayout) -> None:
    if a.n_stages != b.n_stages:
        raise UnalignedLayout(
            f"layouts {a.name!r} ({a.n_stages} stages) and {b.name!r} "
            f"({b.n_stages} stages) cannot be switched between: a switch moves "
            "a boundary, it does not add or remove a rank."
        )
    if a.n_layers != b.n_layers:
        raise UnalignedLayout(
            f"layouts {a.name!r} ({a.n_layers} layers) and {b.name!r} "
            f"({b.n_layers} layers) describe different models."
        )


# ---------------------------------------------------------------------------
# Chunk bands: the unit in which VRAM is actually given back.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BandGeometry:
    """The weight-chunk band geometry, mirroring ``weight_chunk_tag``.

    ``layers_per_chunk``/``chunk_count`` are the two envs
    ``weight_chunk_geometry`` reads (``weg2_memory_saver.py:2304-2313``); the
    tag formula is that function's, not a second copy of the idea -- it is
    reproduced here (and checked against it in the tests) because this module
    must be importable and testable without a CUDA context or a live region.
    """

    layers_per_chunk: int
    chunk_count: int

    def __post_init__(self) -> None:
        if int(self.layers_per_chunk) <= 0 or int(self.chunk_count) <= 0:
            raise UnalignedLayout(
                f"band geometry {self.layers_per_chunk}x{self.chunk_count} is "
                "chunking-OFF. With no chunk tags the memory saver pauses the "
                "weights family as ONE tag, so a boundary move frees nothing at "
                "all and the switch has no motive. Set the chunk envs, or do "
                "not switch."
            )

    @classmethod
    def from_env(cls) -> Optional["BandGeometry"]:
        """The live geometry, or None when chunking is off."""
        from sglang.srt.managers.weg2_memory_saver import weight_chunk_geometry

        layers, count = weight_chunk_geometry()
        if layers <= 0 or count <= 0:
            return None
        return cls(layers_per_chunk=int(layers), chunk_count=int(count))

    def band_of(self, layer_id: int) -> int:
        """``layer_id // layers_per_chunk``, clamped -- ``weight_chunk_tag``."""
        return min(
            int(layer_id) // int(self.layers_per_chunk), int(self.chunk_count) - 1
        )

    def tag_of(self, layer_id: int) -> str:
        return f"weights_{self.band_of(layer_id)}"

    def bands_touched(self, layers: Iterable[int]) -> FrozenSet[int]:
        return frozenset(self.band_of(i) for i in layers)


def bands_freed(
    frm: PLayout, to: PLayout, stage: int, geom: BandGeometry
) -> Tuple[int, ...]:
    """Bands ``stage`` held under ``frm`` and holds NO layer of under ``to``.

    The memory saver pauses whole TAGS, so a band the stage still holds one
    layer of stays resident in full.  This is the ONLY set of bands whose pause
    gives VRAM back, and it is why a boundary that moves seven layers can free
    nothing while one that moves eight frees a whole band.
    """
    held_before = geom.bands_touched(frm.layers_of(stage))
    held_after = geom.bands_touched(to.layers_of(stage))
    return tuple(sorted(held_before - held_after))


def vram_freed_bytes(
    frm: PLayout,
    to: PLayout,
    stage: int,
    geom: BandGeometry,
    layer_bytes: "LayerBytes",
) -> int:
    """Bytes ``stage`` actually gives back moving ``frm`` -> ``to``.

    Sums the WHOLE band, every layer of it, because that is what the pause
    releases -- not the layers that changed owner.  A switch whose answer here
    is 0 moved bytes for nothing, and :func:`decide` prices it accordingly.
    """
    freed = set(bands_freed(frm, to, stage, geom))
    if not freed:
        return 0
    return sum(
        nbytes
        for layer_id, nbytes in layer_bytes.per_layer.items()
        if geom.band_of(layer_id) in freed
    )


def band_aligned_candidates(
    incumbent: PLayout,
    geom: BandGeometry,
    *,
    max_stage0_layers: int,
    min_layers_per_stage: int = 1,
    name_prefix: str = "lean",
) -> Tuple[PLayout, ...]:
    """Every layout whose stage-0 boundary lands ON a band edge and is smaller.

    The search is deliberately narrow: only stage 0's boundary is capped,
    because stage 0 is the card the switch exists to relieve, and only
    band-aligned stage-0 boundaries are offered, because any other boundary
    frees no whole band (:func:`bands_freed`) and therefore cannot shorten the
    later flip no matter how many bytes it moves.  The downstream boundaries are
    enumerated freely -- they cost link time, not stage-0 VRAM -- and the caller
    ranks the results on :func:`vram_freed_bytes` against prefill time.
    """
    n_layers = incumbent.n_layers
    n_stages = incumbent.n_stages
    per = int(geom.layers_per_chunk)
    lo = max(int(min_layers_per_stage), per)
    out: List[PLayout] = []
    for c0 in range(lo, min(int(max_stage0_layers), n_layers) + 1):
        if c0 % per:
            continue
        if c0 >= incumbent.counts[0]:
            continue
        remaining = n_layers - c0
        if remaining < int(min_layers_per_stage) * (n_stages - 1):
            continue
        for tail in _enumerate_tails(
            remaining, n_stages - 1, int(min_layers_per_stage)
        ):
            out.append(
                PLayout(name=f"{name_prefix}{c0}", counts=(c0,) + tail)
                if n_stages > 1
                else PLayout(name=f"{name_prefix}{c0}", counts=(c0,))
            )
    return tuple(out)


def _enumerate_tails(
    remaining: int, stages: int, min_per_stage: int
) -> List[Tuple[int, ...]]:
    if stages <= 0:
        return [()] if remaining == 0 else []
    if stages == 1:
        return [(remaining,)] if remaining >= min_per_stage else []
    out: List[Tuple[int, ...]] = []
    hi = remaining - min_per_stage * (stages - 1)
    for n in range(min_per_stage, hi + 1):
        for tail in _enumerate_tails(remaining - n, stages - 1, min_per_stage):
            out.append((n,) + tail)
    return out


# ---------------------------------------------------------------------------
# What a layer weighs.  From the manifests, never from a per-layer hand number.
# ---------------------------------------------------------------------------

#: ``model.layers.37.self_attn.qkv_proj.weight`` -> 37.  Anchored on the dotted
#: segment so a parameter merely CONTAINING the word does not match, and so a
#: nested ``...layers.3...`` inside a submodule name cannot be read as the
#: stage's layer id.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_index_of(param_name: str) -> Optional[int]:
    """The decoder layer a parameter belongs to, or None for a stage-invariant
    one (embeddings, the head, the final norm, buffers)."""
    m = _LAYER_RE.search(str(param_name))
    return int(m.group(1)) if m else None


@dataclasses.dataclass(frozen=True)
class LayerBytes:
    """Bytes per decoder layer, plus the bytes that belong to no layer.

    Built from the rank manifests, which record every tensor as THIS rank's
    loader actually materialised it (``xchg_manifest.ManifestPiece``) -- so the
    quantisation scales, the biases and the norms are IN the per-layer number
    rather than forgotten beside it, which is the whole reason the number is
    read off the manifest instead of computed from a layer's nominal shape.
    """

    per_layer: Mapping[int, int]
    stage_invariant_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_layer",
            dict(sorted((int(k), int(v)) for k, v in self.per_layer.items())),
        )

    def of(self, layer_id: int) -> int:
        try:
            return int(self.per_layer[int(layer_id)])
        except KeyError:
            raise MissingLayerBytes(
                f"W121: layer {int(layer_id)} changes owner in this switch but "
                f"the byte inventory covers only layers "
                f"{sorted(self.per_layer)[:1]}..{sorted(self.per_layer)[-1:]}. "
                "A move whose size is unknown cannot be priced, and guessing it "
                "from a neighbour is how a per-layer hand number gets back in. "
                "Join the manifests of every stage before planning a switch."
            ) from None

    def total_of(self, layers: Iterable[int]) -> int:
        return sum(self.of(i) for i in layers)

    @property
    def n_layers(self) -> int:
        return len(self.per_layer)

    def digest(self) -> str:
        """A stable fingerprint of the inventory, for the rank-uniformity check."""
        blob = json.dumps(
            {"per_layer": self.per_layer, "invariant": int(self.stage_invariant_bytes)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def layer_bytes_from_manifests(manifests: Iterable[object]) -> LayerBytes:
    """Sum every manifest's pieces into a per-layer byte inventory.

    Takes the JOINED set (every stage's manifest), because a switch prices
    layers this rank does not own: the destination has to know what is coming
    before it arrives.  A layer that appears on two stages -- which under pure
    PP it never should -- is SUMMED rather than max'd or silently deduplicated,
    so the double shows up as a number twice too large instead of as nothing.
    """
    per: Dict[int, int] = {}
    invariant = 0
    for manifest in manifests:
        pieces = getattr(manifest, "pieces", manifest)
        for piece in pieces:
            name = getattr(piece, "param_name", None)
            if name is None:
                continue
            nbytes = int(getattr(piece, "nbytes", 0) or 0)
            idx = layer_index_of(name)
            if idx is None:
                invariant += nbytes
            else:
                per[idx] = per.get(idx, 0) + nbytes
    return LayerBytes(per_layer=per, stage_invariant_bytes=invariant)


# ---------------------------------------------------------------------------
# What a link costs.  Measured pairs only.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LinkRates:
    """Measured card-to-card rates, in decimal GB/s, per ORDERED stage pair.

    ORDERED because the rig is not symmetric and the asymmetry is measured:
    PLAN_BAR1_LANES_0918 records 3080-x8 -> 5090 at 13.15 GB/s and 5090 ->
    3080-x8 at 14.25, 3080-x4 -> 5090 at 6.56 and back at 7.13.  A table keyed
    by an unordered pair would have to pick one of each and would be wrong in
    one direction by up to 9 %.

    A pair that was never measured is a REFUSAL, not an interpolation: the
    x4/x8 spread on this rig is better than 2x, so a guessed rate is not a small
    error, it is the difference between a switch that pays and one that does
    not.
    """

    gbytes_per_s: Mapping[Tuple[int, int], float]

    def __post_init__(self) -> None:
        clean: Dict[Tuple[int, int], float] = {}
        for (src, dst), rate in dict(self.gbytes_per_s).items():
            if float(rate) <= 0:
                raise UnpricedLink(
                    f"W120: link {int(src)}->{int(dst)} is recorded at "
                    f"{rate} GB/s. A non-positive rate is not a slow link, it "
                    "is a missing measurement."
                )
            clean[(int(src), int(dst))] = float(rate)
        object.__setattr__(self, "gbytes_per_s", clean)

    def rate(self, src: int, dst: int) -> float:
        key = (int(src), int(dst))
        if key not in self.gbytes_per_s:
            raise UnpricedLink(
                f"W120: no measured rate for stage {int(src)} -> stage "
                f"{int(dst)}. Measured pairs are "
                f"{sorted(self.gbytes_per_s)}. Refusing to interpolate: the "
                "x4/x8 spread on this rig is better than 2x, so a guessed rate "
                "decides the switch rather than informing it."
            )
        return self.gbytes_per_s[key]

    def seconds(self, src: int, dst: int, nbytes: int) -> float:
        if int(nbytes) <= 0:
            return 0.0
        return float(nbytes) / (self.rate(src, dst) * 1e9)


# ---------------------------------------------------------------------------
# The move plan.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LayerMove:
    layer_id: int
    src_stage: int
    dst_stage: int
    nbytes: int


@dataclasses.dataclass(frozen=True)
class MovePlan:
    frm: PLayout
    to: PLayout
    moves: Tuple[LayerMove, ...]

    @property
    def total_bytes(self) -> int:
        return sum(int(m.nbytes) for m in self.moves)

    def by_pair(self) -> Dict[Tuple[int, int], int]:
        acc: Dict[Tuple[int, int], int] = {}
        for m in self.moves:
            key = (int(m.src_stage), int(m.dst_stage))
            acc[key] = acc.get(key, 0) + int(m.nbytes)
        return dict(sorted(acc.items()))

    def layers_gained_by(self, stage: int) -> Tuple[int, ...]:
        return tuple(m.layer_id for m in self.moves if m.dst_stage == int(stage))

    def layers_lost_by(self, stage: int) -> Tuple[int, ...]:
        return tuple(m.layer_id for m in self.moves if m.src_stage == int(stage))


def plan_layer_moves(frm: PLayout, to: PLayout, layer_bytes: LayerBytes) -> MovePlan:
    """Every layer whose OWNER changes, with the bytes that must follow it.

    Only ownership is compared, never the boundary arithmetic: under a
    three-stage map a single boundary move can cascade (stage 1 both gains
    layers from stage 0 and loses layers to stage 2), and a plan derived from
    "the boundary moved by k" would miss the second half of that.
    """
    _require_same_shape(frm, to)
    moves: List[LayerMove] = []
    for layer_id in range(frm.n_layers):
        src = frm.owner_of(layer_id)
        dst = to.owner_of(layer_id)
        if src == dst:
            continue
        moves.append(
            LayerMove(
                layer_id=layer_id,
                src_stage=src,
                dst_stage=dst,
                nbytes=layer_bytes.of(layer_id),
            )
        )
    return MovePlan(frm=frm, to=to, moves=tuple(moves))


def move_seconds(plan: MovePlan, rates: LinkRates) -> float:
    """Wall time of the move, as the BUSIEST link direction.

    THE MODEL, stated because it is an assumption and not a measurement: each
    card has ONE PCIe link, that link is full duplex, and every peer a card
    talks to in the same direction SHARES it.  So a card's outbound time is the
    SUM over its destinations (they queue on one link) and its inbound time is
    the sum over its sources, while outbound and inbound run CONCURRENTLY
    (duplex) and different cards run concurrently.  The move takes the worst of
    those per-direction loads.

    The duplex half is measured, not assumed: PLAN_BAR1_LANES_0918 records
    3080-x8 <-> 5090 simultaneously at 12.06 + 4.47 GB/s, i.e. duplex works but
    the reverse direction is NOT free.  This model therefore OVERSTATES the
    parallel case and understates nothing; an optimistic move cost would make
    the switch look cheaper than it is, which is the error direction that
    actually costs a window.
    """
    per_pair = plan.by_pair()
    out_s: Dict[int, float] = {}
    in_s: Dict[int, float] = {}
    for (src, dst), nbytes in per_pair.items():
        t = rates.seconds(src, dst, nbytes)
        out_s[src] = out_s.get(src, 0.0) + t
        in_s[dst] = in_s.get(dst, 0.0) + t
    return max([0.0] + list(out_s.values()) + list(in_s.values()))


# ---------------------------------------------------------------------------
# Prefill cost of a layout, from the calibrated per-stage timing.
# ---------------------------------------------------------------------------


def prefill_seconds(
    layout: PLayout, pending_tokens: int, calib: "SwitchCalibration"
) -> float:
    """Seconds to prefill ``pending_tokens`` under ``layout``.

    Priced in SECONDS throughout rather than in tok/s, which removes the
    mixed-unit trap ``launcher.derive_x_star`` has to refuse by name (#1271): a
    group-throughput rate and a request-latency rate are not the same quantity,
    and a rule that divides one by the other produces a number with no meaning.
    Here both layouts are priced through ONE ``PrefillTiming`` object, so the
    two sides of every comparison share a measurement basis by construction and
    there is no second unit to mix.

    Chunk quantisation is kept: a partial chunk costs a whole chunk, because the
    pipeline is charged per chunk and pretending otherwise flatters the fast
    layout on exactly the small backlogs where the rule is closest to its
    break-even.
    """
    from sglang.srt.planner.pp_cut import pipelined_prefill_ms

    if int(pending_tokens) <= 0:
        return 0.0
    chunks = math.ceil(int(pending_tokens) / int(calib.chunk_tokens))
    per_chunk_s = float(pipelined_prefill_ms(layout.counts, calib.timing)) / 1000.0
    return chunks * per_chunk_s


def _per_token_seconds(layout: PLayout, calib: "SwitchCalibration") -> float:
    """The continuous per-token cost, for the break-even solve only."""
    from sglang.srt.planner.pp_cut import pipelined_prefill_ms

    per_chunk_s = float(pipelined_prefill_ms(layout.counts, calib.timing)) / 1000.0
    return per_chunk_s / float(calib.chunk_tokens)


# ---------------------------------------------------------------------------
# The calibration bundle and the rule.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SwitchCalibration:
    """Every measured input the rule needs. No field has a default that invents
    a number; ``margin_s`` defaults to 0 and the verdict SAYS it is unguarded."""

    #: ``planner.pp_cut.PrefillTiming``. ONE object for both layouts.
    timing: object
    #: D's prefill chunk width -- the quantum a backlog is charged in.
    chunk_tokens: int
    rates: LinkRates
    layer_bytes: LayerBytes
    geom: BandGeometry
    #: Measured P->D flip seconds, per layout NAME. From the boot's own
    #: ``flip_total=`` lines (``launcher._RE_FLIP``), never predicted here.
    flip_seconds: Mapping[str, float]
    #: The deadband. A switch must beat staying by MORE than this. It is the
    #: caller's own measurement noise (e.g. the spread of its per-chunk stage
    #: times), not a taste setting; 0.0 means unguarded and is reported as such,
    #: because inventing a margin is inventing a number.
    margin_s: float = 0.0

    def __post_init__(self) -> None:
        if int(self.chunk_tokens) <= 0:
            raise PLayoutSwitchError(
                f"chunk_tokens={self.chunk_tokens}: a backlog cannot be divided "
                "into chunks of zero tokens."
            )
        object.__setattr__(self, "flip_seconds", dict(self.flip_seconds))
        if float(self.margin_s) < 0:
            raise PLayoutSwitchError(
                f"margin_s={self.margin_s} is negative, which would make a "
                "switch that LOSES time acceptable."
            )

    def flip_of(self, layout: PLayout) -> float:
        try:
            return float(self.flip_seconds[layout.name])
        except KeyError:
            raise UnpricedFlip(
                f"W122: no measured flip time for layout {layout.name!r}; "
                f"priced layouts are {sorted(self.flip_seconds)}. The whole "
                "rule is 'is the faster prefill worth the longer flip', so a "
                "layout with no flip time of its own cannot be one of its "
                "arguments. Measure it (the boot's flip_total= lines) or do not "
                "offer the layout."
            ) from None


ACTION_STAY = "stay"
ACTION_TO_FAST = "to_fast"
ACTION_TO_LEAN = "to_lean"
ACTION_FAST_THEN_LEAN = "fast_then_lean"
ACTIONS = (ACTION_STAY, ACTION_TO_FAST, ACTION_TO_LEAN, ACTION_FAST_THEN_LEAN)


@dataclasses.dataclass(frozen=True)
class Plan:
    """One priced course of action, end to end: switch, prefill, flip."""

    action: str
    end_layout: str
    switch_s: float
    prefill_s: float
    back_switch_s: float
    flip_s: float
    moved_bytes: int
    vram_freed_bytes: int

    @property
    def total_s(self) -> float:
        return (
            float(self.switch_s)
            + float(self.prefill_s)
            + float(self.back_switch_s)
            + float(self.flip_s)
        )


@dataclasses.dataclass(frozen=True)
class SwitchVerdict:
    """PP0's single answer, and the numbers it was reached from."""

    action: str
    pending_tokens: int
    chosen: Plan
    incumbent: Plan
    plans: Tuple[Plan, ...]
    gain_s: float
    margin_s: float
    breakeven_tokens: Optional[int]
    why: str
    #: The union's token price per stage, when it was priced. Reported rather
    #: than folded into a second: tokens and seconds are different units.
    union_costs: Tuple["UnionKvCost", ...] = ()

    @property
    def switches(self) -> bool:
        return self.action != ACTION_STAY

    def as_line(self) -> str:
        """One log line. Every figure names its own basis."""
        be = "none" if self.breakeven_tokens is None else str(self.breakeven_tokens)
        return (
            f"WEG2-PSWITCH action={self.action} pending={self.pending_tokens} "
            f"from={self.incumbent.end_layout} to={self.chosen.end_layout} "
            f"stay_s={self.incumbent.total_s:.3f} best_s={self.chosen.total_s:.3f} "
            f"gain_s={self.gain_s:.3f} margin_s={self.margin_s:.3f} "
            f"margin={'unguarded' if self.margin_s <= 0 else 'guarded'} "
            f"switch_s={self.chosen.switch_s:.3f} "
            f"back_s={self.chosen.back_switch_s:.3f} "
            f"prefill_s={self.chosen.prefill_s:.3f} flip_s={self.chosen.flip_s:.3f} "
            f"moved_mib={self.chosen.moved_bytes / 1048576:.0f} "
            f"freed_mib={self.chosen.vram_freed_bytes / 1048576:.0f} "
            f"breakeven_tokens={be} "
            f"union_tok_lost={max((c.tokens_lost for c in self.union_costs), default=0)} "
            f"why={self.why}"
        )


def _price(
    action: str,
    *,
    current: PLayout,
    run_on: PLayout,
    end_on: PLayout,
    pending_tokens: int,
    calib: SwitchCalibration,
    recapture_s: float = 0.0,
) -> Plan:
    """Price one course: switch to ``run_on``, prefill there, end on ``end_on``.

    ``recapture_s`` is the CUDA-graph re-record the switch forces, charged to
    each LEG that actually moves the boundary. It is a real cost on the
    critical path, not an aside: the captured prefill graphs bake the executed
    layer set, so a boundary that moves leaves every graph to be re-recorded or
    discarded (``runner/graph_recapture.py``). Zero by default so a caller who
    has not measured it gets the old arithmetic rather than a guessed term.
    """
    switch_plan = plan_layer_moves(current, run_on, calib.layer_bytes)
    back_plan = plan_layer_moves(run_on, end_on, calib.layer_bytes)
    freed = vram_freed_bytes(current, end_on, 0, calib.geom, calib.layer_bytes)
    recap = float(recapture_s)
    moves_out = current.counts != run_on.counts
    moves_back = run_on.counts != end_on.counts
    return Plan(
        action=action,
        end_layout=end_on.name,
        switch_s=move_seconds(switch_plan, calib.rates) + (recap if moves_out else 0.0),
        prefill_s=prefill_seconds(run_on, pending_tokens, calib),
        back_switch_s=(
            move_seconds(back_plan, calib.rates) + (recap if moves_back else 0.0)
        ),
        flip_s=calib.flip_of(end_on),
        moved_bytes=switch_plan.total_bytes + back_plan.total_bytes,
        vram_freed_bytes=freed,
    )


def course_breakeven_tokens(
    current: PLayout,
    run_on: PLayout,
    end_on: PLayout,
    calib: SwitchCalibration,
) -> Optional[int]:
    """Backlog at which running on ``run_on`` and ending on ``end_on`` beats
    staying on ``current``.

    ``N* = (switch_s + back_switch_s + flip(end_on) - flip(current))
           / (s_current - s_run_on)``,
    the same shape as ``launcher.derive_x_star`` and with the same refusal: a
    layout that is not FASTER per token has no break-even, and answering one
    anyway is the confident-wrong class.  ``None`` means "never pays", which is
    a real answer and is reported as one.

    PER COURSE, not per layout, and that distinction is not cosmetic.  The
    ``fast_then_lean`` course does not pay the flip DELTA -- it ends on the lean
    layout -- so its numerator is smaller and it starts paying at a SMALLER
    backlog than the pure ``to_fast`` course does.  A single "break-even of the
    fast layout" would therefore be wrong for the course the rule actually
    picks, and would read as a bug the first time :func:`decide` switched below
    it.  (It did: this function was two-argument in its first draft and a test
    comparing the two numbers caught it.)

    Solved on the CONTINUOUS per-token cost, so it is the ideal crossing and the
    chunk-quantised :func:`decide` may switch one chunk later.  Stated because a
    reader comparing the two numbers must not read that gap as a bug either.
    """
    s_cur = _per_token_seconds(current, calib)
    s_run = _per_token_seconds(run_on, calib)
    denom = s_cur - s_run
    if denom <= 0:
        return None
    numer = (
        move_seconds(plan_layer_moves(current, run_on, calib.layer_bytes), calib.rates)
        + move_seconds(plan_layer_moves(run_on, end_on, calib.layer_bytes), calib.rates)
        + calib.flip_of(end_on)
        - calib.flip_of(current)
    )
    if numer <= 0:
        return 0
    return int(math.ceil(numer / denom))


def breakeven_tokens(
    current: PLayout, fast: PLayout, calib: SwitchCalibration
) -> Optional[int]:
    """The pure ``to_fast`` course's break-even. See
    :func:`course_breakeven_tokens`, of which this is the two-layout case."""
    return course_breakeven_tokens(current, fast, fast, calib)


def decide(
    *,
    current: PLayout,
    fast: PLayout,
    lean: PLayout,
    pending_tokens: int,
    calib: SwitchCalibration,
    union_costs: Optional[Sequence["UnionKvCost"]] = None,
    required_context_tokens: Optional[int] = None,
    recapture_s: float = 0.0,
) -> SwitchVerdict:
    """PP0's verdict: which of the four courses finishes soonest.

    THE RULE the order asks for, in the form it asks for it.  Every course is
    priced end to end -- the switch itself, the prefill of the backlog that is
    outstanding RIGHT NOW, and the P->D flip that follows -- and the shortest
    wins.  That is the same trade as "is the tok/s gain over the pending tokens
    bigger than the longer flip", with two things the phrasing leaves implicit
    made explicit because they change the answer:

    * the switch is not free, so it is a term and not a footnote;
    * ``fast_then_lean`` exists.  Running the backlog on the fast layout and
      moving BACK to the lean one before the flip buys the fast prefill without
      the long flip, at the price of a second move.  Over a big backlog it beats
      both pure courses, and a rule offering only "stay" and "go fast" would
      never find it.

    RE-EVALUATED, not latched.  The caller runs this at every chunk boundary and
    on every new arrival, with ``pending_tokens`` as it stands then and
    ``current`` as the layout actually in force -- so a backlog that grows while
    P is already prefilling can still tip the decision, which is exactly the
    case the order opens with ("es koennen ja waehrend des laufenden prefills
    noch weitere pending dazukommen").  Because every course is priced from NOW
    and the switch cost is in the comparison, each verdict is myopically optimal
    and a reversal only happens when the backlog genuinely moved.

    ``margin_s`` is the deadband against deciding on noise. It is the caller's
    own measured spread, and when it is 0 the verdict says ``unguarded`` rather
    than pretending to a confidence it has not got.
    """
    _require_same_shape(current, fast)
    _require_same_shape(current, lean)
    pending = max(0, int(pending_tokens))

    # The union's KV price, checked BEFORE any course is priced. Offering a
    # layout pair whose union cannot hold the promised context and then picking
    # the quickest course through it would be answering the wrong question
    # quickly: the pools cover the union for the whole life of the boot, so a
    # union that is too wide is a property of the PAIR and is settled here,
    # once, not re-litigated per verdict.
    if union_costs is not None and required_context_tokens is not None:
        require_union_fits(union_costs, int(required_context_tokens))

    stay = _price(
        ACTION_STAY,
        current=current,
        run_on=current,
        end_on=current,
        pending_tokens=pending,
        calib=calib,
        recapture_s=recapture_s,
    )
    plans: List[Plan] = [stay]
    if fast.counts != current.counts:
        plans.append(
            _price(
                ACTION_TO_FAST,
                current=current,
                run_on=fast,
                end_on=fast,
                pending_tokens=pending,
                calib=calib,
                recapture_s=recapture_s,
            )
        )
    if lean.counts != current.counts:
        plans.append(
            _price(
                ACTION_TO_LEAN,
                current=current,
                run_on=lean,
                end_on=lean,
                pending_tokens=pending,
                calib=calib,
                recapture_s=recapture_s,
            )
        )
    if fast.counts != lean.counts:
        plans.append(
            _price(
                ACTION_FAST_THEN_LEAN,
                current=current,
                run_on=fast,
                end_on=lean,
                pending_tokens=pending,
                calib=calib,
                recapture_s=recapture_s,
            )
        )

    best = min(plans, key=lambda p: (p.total_s, ACTIONS.index(p.action)))
    gain = stay.total_s - best.total_s
    margin = float(calib.margin_s)
    if best.action == ACTION_STAY:
        why = "staying is already the shortest course"
        chosen = stay
        action = ACTION_STAY
    elif gain <= margin:
        why = (
            f"best course {best.action!r} saves {gain:.3f} s, which does not "
            f"clear the {margin:.3f} s deadband"
        )
        chosen = stay
        action = ACTION_STAY
    else:
        why = f"{best.action!r} finishes {gain:.3f} s sooner than staying"
        chosen = best
        action = best.action

    # THE EARLIEST backlog at which this stops being 'stay' -- the minimum over
    # the courses actually on offer, not the pure to_fast one. A scheduler uses
    # this to know when it is worth asking again, so a number larger than the
    # one the rule acts on would make it ask too late.
    candidates = [
        course_breakeven_tokens(current, run_on, end_on, calib)
        for run_on, end_on in (
            (fast, fast),
            (lean, lean),
            (fast, lean),
        )
        if run_on.counts != current.counts or end_on.counts != current.counts
    ]
    live = [n for n in candidates if n is not None]

    return SwitchVerdict(
        action=action,
        pending_tokens=pending,
        chosen=chosen,
        incumbent=stay,
        plans=tuple(plans),
        gain_s=gain,
        margin_s=margin,
        breakeven_tokens=min(live) if live else None,
        why=why,
        union_costs=tuple(union_costs or ()),
    )


# ---------------------------------------------------------------------------
# Quiescence: a switch happens at a chunk boundary, on a drained ring.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RingState:
    """What the PP ring carries right now."""

    at_chunk_boundary: bool
    inflight_microbatches: int
    inflight_requests: int


def require_switchable(ring: RingState) -> None:
    """Refuse a switch while the ring still carries work.

    Three separate reasons, all fatal, so the refusal names which one:

    * IN-FLIGHT ACTIVATIONS.  A microbatch in the ring was sent from a stage
      boundary that is about to move.  ``PPProxyTensors``
      (``forward_batch_info.py:1593``) carries it to a receiver chosen by the
      OLD map; after the move that receiver no longer owns the layer the
      activation is for.
    * RECURRENT STATE.  A GDN layer's per-sequence state lives WITH the layer
      (~19.5 MiB temporal + 0.762 conv per layer) and this switch does not move
      it -- same scope line #704's actuator draws, and its ``quiescent`` flag
      refuses for exactly this reason.
    * MID-CHUNK.  A boundary that moves inside a chunk splits one forward across
      two maps, which is the silent-wrong shape rather than the loud one.
    """
    if not ring.at_chunk_boundary:
        raise SwitchNotQuiescent(
            "W124: not at a chunk boundary. A layer boundary that moves inside "
            "a chunk splits one forward pass across two stage maps -- the first "
            "half computed by the old owner, the second by nobody -- which "
            "produces plausible output from a shallower model rather than an "
            "error."
        )
    if int(ring.inflight_microbatches) > 0:
        raise SwitchNotQuiescent(
            f"W124: {int(ring.inflight_microbatches)} microbatch(es) still in "
            "the PP ring. Their activations were sent to receivers chosen by "
            "the OLD stage map; after the move those receivers no longer own "
            "the layers the activations are for. Drain the ring first."
        )
    if int(ring.inflight_requests) > 0:
        raise SwitchNotQuiescent(
            f"W124: {int(ring.inflight_requests)} request(s) still in flight. A "
            "GDN layer's per-sequence recurrent state travels with the layer "
            "and this switch moves weights only, so an in-flight sequence would "
            "continue against state that is no longer on its card."
        )


# ---------------------------------------------------------------------------
# Rank uniformity: ONE verdict, taken at PP0, checked everywhere.
# ---------------------------------------------------------------------------


def preconditions_digest(
    *,
    current: PLayout,
    fast: PLayout,
    lean: PLayout,
    geom: BandGeometry,
    layer_bytes: LayerBytes,
    epoch: int,
    chunk_index: int,
) -> str:
    """A fingerprint of everything a rank must already agree on.

    Deliberately does NOT cover the pending-token count, the link rates or the
    flip times: only PP0 sees the queue and only PP0 prices the courses (#968 --
    PP0 is authoritative, downstream is verdict-free).  Putting them in would
    force every rank to hold a copy of PP0's inputs, and a rank holding a copy
    of an input is a rank that can derive a second verdict from it, which is the
    split-brain this check exists to prevent rather than to enable.

    What it DOES cover is the shared state a switch is only meaningful against:
    the three layouts, the band geometry, the byte inventory, and WHICH switch
    point this is.  The last two entries are what stop a verdict from a previous
    chunk being applied to this one.
    """
    blob = json.dumps(
        {
            "current": [current.name, list(current.counts)],
            "fast": [fast.name, list(fast.counts)],
            "lean": [lean.name, list(lean.counts)],
            "geom": [int(geom.layers_per_chunk), int(geom.chunk_count)],
            "bytes": layer_bytes.digest(),
            "epoch": int(epoch),
            "chunk_index": int(chunk_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def pp0_broadcast(verdict: SwitchVerdict, digest: str) -> Dict[str, object]:
    """The record PP0 puts on the wire. Small on purpose: an ACTION and the
    PRECONDITIONS it was decided under, never the reasoning."""
    return {
        "action": str(verdict.action),
        "to": str(verdict.chosen.end_layout),
        "run_on": (
            str(verdict.chosen.end_layout)
            if verdict.action != ACTION_FAST_THEN_LEAN
            else None
        ),
        "digest": str(digest),
        "pending_tokens": int(verdict.pending_tokens),
        "gain_s": round(float(verdict.gain_s), 6),
    }


def accept_pp0_verdict(
    record: Mapping[str, object], local_digest: str, local_rank: int
) -> str:
    """Downstream's ONLY move: check the preconditions, then obey.

    #968 to the letter -- a downstream stage does not re-derive the verdict and
    is given nothing to re-derive it from.  It checks that PP0 decided under the
    SAME preconditions this rank is holding, and on a mismatch it raises on
    EVERY rank rather than skipping its own switch.

    Why a mismatch cannot be a local skip: the layouts tile ``[0, n_layers)``
    exactly (``layout_boundary.validate_world_tiling``), so a stage that does
    not follow leaves layers owned twice (computed twice) or owned by nobody
    (silently skipped, producing plausible output from a shallower model). One
    stage out of step is strictly worse than no switch at all.
    """
    action = record.get("action")
    if action not in ACTIONS:
        raise PLayoutRankDisagree(
            f"W125: rank {int(local_rank)} received action {action!r}, which is "
            f"not one of {list(ACTIONS)}. A stage cannot act on a verdict it "
            "cannot name, and guessing 'stay' would leave the other stages "
            "mid-switch."
        )
    got = str(record.get("digest", ""))
    if got != str(local_digest):
        raise PLayoutRankDisagree(
            f"W125: rank {int(local_rank)} holds preconditions "
            f"{str(local_digest)!r} but PP0 decided under {got!r}. The ranks "
            "disagree about the layouts, the band geometry, the byte inventory "
            "or WHICH switch point this is. STOP: a switch applied on some "
            "stages and not others leaves layers owned twice or owned by "
            "nobody, and the second of those is silent."
        )
    return str(action)


# ---------------------------------------------------------------------------
# The mover: deposit / collect, over a BAR1 peer window.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoveOp:
    """One ``memcpy_async`` the switch will issue, named before it is issued.

    ``kind`` is one of:

    * ``deposit``  -- sender writes its layer into the receiver's BAR1 window;
    * ``collect``  -- receiver copies from its own window into the layer region;
    * ``direct``   -- sender writes STRAIGHT into the receiver's layer region,
      no collect at all. This is the 5090 case: its BAR1 aperture covers its
      whole VRAM (32 GiB BAR), so the region itself is the window
      (PLAN_BAR1_LANES_0918, "kein Collect"). The 3080s cannot do this -- 256
      MiB of BAR1 -- and take the two-step path.
    """

    kind: str
    layer_id: int
    dst: int
    src: int
    nbytes: int
    peer_stage: int


def emit_move_ops(plan: MovePlan, regions, this_stage: int) -> Tuple[MoveOp, ...]:
    """The ops THIS stage issues for ``plan``, in layer order.

    ``regions`` is injected rather than reached for, which is what makes the
    whole mover testable without a card. It must offer:

    * ``src_ptr(layer_id) -> int``            -- this rank's layer region;
    * ``dst_ptr(layer_id) -> int``            -- this rank's destination region;
    * ``window(src_stage, dst_stage) -> obj`` -- the mapped peer window, with
      ``.dev_ptr`` (int) and ``.direct`` (bool); ``None`` for a pair with no
      window, which is a refusal here rather than a silent host fallback.

    The shapes mirror the flip legs exactly: deposit is
    ``memcpy_async(base + slot_off, src_ptr, n, stream)``
    (``weight_exchange_bounce.py:917``), collect is
    ``memcpy_async(dst_ptr, base + slot_off, n, stream)`` (``:941``). The
    difference is only that ``base`` is a peer BAR1 window rather than a
    host-registered buffer, so no byte crosses host DRAM.
    """
    ops: List[MoveOp] = []
    for m in plan.moves:
        if m.src_stage == int(this_stage):
            win = regions.window(m.src_stage, m.dst_stage)
            if win is None:
                raise UnpricedLink(
                    f"W120: no BAR1 window for stage {m.src_stage} -> "
                    f"{m.dst_stage}, so layer {m.layer_id} has no route. "
                    "Refusing rather than falling back to the host bounce "
                    "unannounced: the host path is ~3x slower on this rig and a "
                    "switch priced on the BAR1 rate would then lose time it was "
                    "predicted to save."
                )
            ops.append(
                MoveOp(
                    kind="direct" if getattr(win, "direct", False) else "deposit",
                    layer_id=m.layer_id,
                    dst=int(win.dev_ptr),
                    src=int(regions.src_ptr(m.layer_id)),
                    nbytes=int(m.nbytes),
                    peer_stage=int(m.dst_stage),
                )
            )
        elif m.dst_stage == int(this_stage):
            win = regions.window(m.src_stage, m.dst_stage)
            if win is None:
                raise UnpricedLink(
                    f"W120: no BAR1 window for stage {m.src_stage} -> "
                    f"{m.dst_stage}; this rank cannot receive layer "
                    f"{m.layer_id}."
                )
            if getattr(win, "direct", False):
                # The sender wrote into the region itself. Nothing to collect,
                # and issuing a region->region copy here would be a real bug:
                # src and dst would be the same address.
                continue
            ops.append(
                MoveOp(
                    kind="collect",
                    layer_id=m.layer_id,
                    dst=int(regions.dst_ptr(m.layer_id)),
                    src=int(win.dev_ptr),
                    nbytes=int(m.nbytes),
                    peer_stage=int(m.src_stage),
                )
            )
    return tuple(ops)


def run_move_ops(ops: Iterable[MoveOp], device_ops, stream: int) -> int:
    """Issue the ops on ``stream``; return the bytes issued.

    ``device_ops`` is a ``weight_exchange_transport.DeviceOps``; only
    ``memcpy_async(dst, src, nbytes, stream)`` is used, which is the one
    primitive every flip leg already goes through.
    """
    moved = 0
    for op in ops:
        device_ops.memcpy_async(int(op.dst), int(op.src), int(op.nbytes), int(stream))
        moved += int(op.nbytes)
    return moved


# ---------------------------------------------------------------------------
# The union's KV price (blocker 3).
#
# A layer this rank may be handed by a later switch needs KV rows on the card
# BEFORE the switch, not after it: the pools are built once, over the UNION of
# the layouts, because rebuilding one at a new start_layer would move the
# `layer_id - start_layer` indexing base under every row already cached
# (layout_boundary.pool_coverage_observer). That is not free. The pool's token
# capacity is `kv_budget_bytes // cell_size` and the cell scales with the layer
# count (pool_configurator.py:408, :555), so holding U layers instead of N
# costs tokens on every card -- and a switch whose prefill gain is paid for
# with context the user asked for is not a gain.
# ---------------------------------------------------------------------------


class UnionKvTooCostly(PLayoutSwitchError):
    """The union's pool would hold fewer tokens than the run requires."""


@dataclasses.dataclass(frozen=True)
class UnionKvCost:
    """Pool tokens a card loses by holding the union instead of one layout.

    Deliberately NOT expressed in seconds. Tokens and seconds are different
    units and the honest conversion needs a measured recompute rate for evicted
    prefix; inventing one here would put a fabricated term in ``decide`` beside
    the measured ones, which is exactly the mixed-unit class ``derive_x_star``
    refuses by name. So this is a CONSTRAINT and a reported number: the union
    either leaves each card enough tokens for the context the run promises, or
    the pair of layouts is refused and the switch is not offered at all.
    """

    stage: int
    #: KV bytes this card gives the pool, after weights and reserve.
    kv_budget_bytes: int
    #: Bytes one token costs in ONE layer's KV (per_token_bytes / layer_count).
    bytes_per_token_per_layer: int
    #: Layers this stage runs under the layout being priced.
    layout_layers: int
    #: Layers the union makes it hold rows for.
    union_layers: int

    def __post_init__(self) -> None:
        for name in ("kv_budget_bytes", "bytes_per_token_per_layer"):
            if int(getattr(self, name)) <= 0:
                raise UnionKvTooCostly(
                    f"{name}={getattr(self, name)} on stage {self.stage}: a "
                    "card with no KV budget and a token that costs no bytes "
                    "are both numbers that were never measured, and the token "
                    "counts below would be arithmetic on a guess."
                )
        if int(self.union_layers) < int(self.layout_layers):
            raise UnionKvTooCostly(
                f"stage {self.stage}: the union holds {self.union_layers} "
                f"layers but the layout runs {self.layout_layers}. The union is "
                "the span of every layout this rank may occupy, so it can never "
                "be the smaller of the two -- this is a union computed from the "
                "wrong set of layouts."
            )

    @property
    def tokens_with_layout(self) -> int:
        """Pool tokens if the pool were built for this layout alone."""
        return int(self.kv_budget_bytes) // (
            int(self.bytes_per_token_per_layer) * int(self.layout_layers)
        )

    @property
    def tokens_with_union(self) -> int:
        """Pool tokens actually available, the pool being built over the union."""
        return int(self.kv_budget_bytes) // (
            int(self.bytes_per_token_per_layer) * int(self.union_layers)
        )

    @property
    def tokens_lost(self) -> int:
        return self.tokens_with_layout - self.tokens_with_union

    def as_line(self) -> str:
        return (
            f"[p-layout-union] stage {self.stage}: union {self.union_layers} "
            f"layers vs layout {self.layout_layers} -> pool "
            f"{self.tokens_with_union} tok (would be {self.tokens_with_layout}), "
            f"costs {self.tokens_lost} tok"
        )


def union_layout(*layouts: PLayout) -> Tuple[Tuple[int, int], ...]:
    """Per stage, the union span every rank must build weights and pools over.

    The span, not the set: :class:`PLayout` is contiguous by construction (the
    gapped form was refused on boot weg2gp1), so the union of a stage's ranges
    across the layouts is an interval and a rank holds every layer in it.
    """
    if not layouts:
        raise PLayoutSwitchError("union_layout needs at least one layout.")
    first = layouts[0]
    for other in layouts[1:]:
        _require_same_shape(first, other)
    spans: List[Tuple[int, int]] = []
    for stage in range(first.n_stages):
        los, his = [], []
        for lay in layouts:
            a, b = lay.range_of(stage)
            los.append(a)
            his.append(b)
        spans.append((min(los), max(his)))
    return tuple(spans)


def union_kv_costs(
    *layouts: PLayout,
    priced_layout: PLayout,
    kv_budget_bytes: Mapping[int, int],
    bytes_per_token_per_layer: Mapping[int, int],
) -> Tuple[UnionKvCost, ...]:
    """The union's token price on every stage, for ``priced_layout``."""
    spans = union_layout(*layouts)
    out: List[UnionKvCost] = []
    for stage, (lo, hi) in enumerate(spans):
        a, b = priced_layout.range_of(stage)
        try:
            budget = kv_budget_bytes[stage]
            per_layer = bytes_per_token_per_layer[stage]
        except KeyError:
            raise UnionKvTooCostly(
                f"stage {stage} has no measured KV budget / per-layer token "
                f"cost; priced stages are {sorted(set(kv_budget_bytes))}. The "
                "cards differ by more than 50 % in VRAM on this rig, so one "
                "number standing in for another card's is not an "
                "approximation -- it is the wrong card."
            ) from None
        out.append(
            UnionKvCost(
                stage=stage,
                kv_budget_bytes=int(budget),
                bytes_per_token_per_layer=int(per_layer),
                layout_layers=b - a,
                union_layers=hi - lo,
            )
        )
    return tuple(out)


def require_union_fits(
    costs: Sequence[UnionKvCost], required_tokens: int
) -> Tuple[UnionKvCost, ...]:
    """Refuse the layout PAIR when the union cannot hold the promised context.

    ``required_tokens`` is the context the run promises -- 262144 under the
    standing order that Next Flash is always computed at 262k. A pool that
    cannot hold it has not made the prefill faster; it has made the run
    smaller, and the switch would be bought with the user's context.
    """
    need = int(required_tokens)
    short = [c for c in costs if c.tokens_with_union < need]
    if short:
        lines = "; ".join(
            f"stage {c.stage}: {c.tokens_with_union} tok over its "
            f"{c.union_layers}-layer union (needs {need}, loses {c.tokens_lost} "
            f"to the union)"
            for c in short
        )
        raise UnionKvTooCostly(
            f"W123: the union of these layouts does not leave room for "
            f"{need} tokens on every card -- {lines}. The pools must cover the "
            "union (a layer a later switch activates has no rows otherwise, and "
            "rebuilding at a new start moves the indexing base under every "
            "cached row), so this is a property of the layout PAIR, not of "
            "either layout. Offer a pair whose union is narrower, or lower the "
            "promised context deliberately."
        )
    return tuple(costs)


# ---------------------------------------------------------------------------
# The mover's route (blocker 4).
# ---------------------------------------------------------------------------


class MoverNotCapturable(PLayoutSwitchError):
    """The mover was asked to run inside a CUDA-graph capture."""


def require_breakable_route(
    *, capturing: bool, route: str, n_ops: int = 0, capture_safety=None
) -> str:
    """The mover runs eager/breakable only. Refused by NAME under capture.

    ``barlink_bar1_p2p.capture_safety()`` already states the asymmetry that
    forces this (``barlink_bar1_p2p.py:181``): ``put()`` is a ``memcpy_async``
    on the caller's stream and records into a graph, but the RECV half has to
    be waited on and BAR1 exposes no device-side wait -- its three kernels are
    all collectives, and a host-side flag spin inside a capture raises
    ``cudaErrorStreamCaptureUnsupported``.

    The failure that matters is not the raise. It is the HALF that does record:
    a capture containing the deposits but not the collects replays as a mover
    that ships bytes into peer windows and never lands them, every replay, in
    silence. So the refusal is placed BEFORE the first op is issued and names
    the route, rather than being left to CUDA to discover midway.
    """
    safety = (capture_safety or _bar1_capture_safety)()
    if not capturing:
        return str(route)
    raise MoverNotCapturable(
        f"W124: the P-layout mover was asked to issue {n_ops} op(s) on route "
        f"{route!r} INSIDE a CUDA-graph capture. BAR1 recv is not capturable: "
        f"{safety.get('reason')}. The deposit half WOULD record, so a captured "
        "mover is not a capture failure at every replay -- it is a mover that "
        "deposits into peer windows and never collects, silently, on the fast "
        "path. Run the switch on the eager/breakable route at a chunk boundary "
        "(require_switchable), which is where it belongs anyway: the ring must "
        "be drained for the switch to be safe at all."
    )


def _bar1_capture_safety():
    from sglang.srt.distributed.device_communicators.barlink_bar1_p2p import (
        capture_safety,
    )

    return capture_safety()


def run_move_ops_guarded(
    ops: Iterable[MoveOp],
    device_ops,
    stream: int,
    *,
    capturing: bool,
    route: str = "breakable",
    capture_safety=None,
) -> int:
    """:func:`run_move_ops` behind :func:`require_breakable_route`."""
    ops = tuple(ops)
    require_breakable_route(
        capturing=capturing,
        route=route,
        n_ops=len(ops),
        capture_safety=capture_safety,
    )
    return run_move_ops(ops, device_ops, stream)


# ---------------------------------------------------------------------------
# Who the 5090's PP neighbour is.
#
# The standing shape on this rig is --rank-gpu-id 0,1,2, i.e. stage 0 = 5090
# (cuda:0), stage 1 = the 3080 on the x4 link, stage 2 = the 3080 on x8
# (planner/device_map.py:36-38, planner/flags.py:2615-2625). That puts the SLOW
# card in the MIDDLE, where it is a neighbour on both hops, and every layer the
# switch moves between stage 0 and stage 1 crosses at 6.56 GB/s instead of
# 13.15 (PLAN_BAR1_LANES_0918).
#
# The PP stage ORDER itself is not free: a stage number is welded to a rank
# number (server_args.py:10700 `world_rank = pp_rank * tp_size + tp_rank`) and
# the PP group is built from a strided range (parallel_state.py:3945-3962), so
# no rank list puts rank 2 at stage 1. The send/recv layer WOULD follow a
# permuted list -- neighbours are positional, `self.ranks[(rank_in_group ± 1)
# % world_size]` (parallel_state.py:1319-1350) -- but nothing constructs one.
#
# The equivalent move IS free, and it is the supported one: permute the CARDS
# across the stages with --rank-gpu-id (server_args.py:10583-10605). `0,2,1`
# makes the 5090's neighbour the x8 3080 and exiles the x4 card to the end of
# the pipeline, where it has one hop instead of two.
#
# The hazard is not the permutation. It is permuting it HALFWAY: at least five
# vectors are indexed by rank and were hand-calibrated for 0,1,2, and each one
# left behind silently gives one card another card's budget. So the permutation
# is applied to all of them at once, by one object, or it is refused.
# ---------------------------------------------------------------------------


class StagePermutationIncomplete(PLayoutSwitchError):
    """A card permutation applied to some rank-indexed vectors but not all."""


#: Every rank-indexed vector that must move with the cards. Named here so the
#: list is a datum a test can assert against, not a habit spread over a
#: briefing. Sources: planner/flags.py:2615-2625 (tokvec, reserve),
#: weg2/launcher.py:2884 (--rank-gpu-memory-mib), server_args.py:17254-17290
#: (--pp-stage-ratio), p_layout_switch.LinkRates (ordered stage pairs).
RANK_INDEXED_VECTORS = (
    "rank_gpu_id",
    "rank_gpu_memory_mib",
    "pp_stage_ratio",
    "kv_tokvec",
    "vram_reserve_mib",
)


@dataclasses.dataclass(frozen=True)
class StagePermutation:
    """Move the cards across the stages, and everything keyed to a card with them.

    ``cards[stage]`` is the physical cuda index that stage runs on. The default
    ``(0, 1, 2)`` is the shipped placement; ``(0, 2, 1)`` is the one that makes
    the 5090's neighbour the x8 3080.
    """

    cards: Tuple[int, ...]

    def __post_init__(self) -> None:
        cards = tuple(int(c) for c in self.cards)
        object.__setattr__(self, "cards", cards)
        if sorted(cards) != list(range(len(cards))):
            raise StagePermutationIncomplete(
                f"cards={list(cards)} is not a permutation of "
                f"{list(range(len(cards)))}. Two stages on one card is refused "
                "upstream by name (server_args.py:10893-10900), and a missing "
                "card is a stage with no GPU."
            )

    @property
    def is_identity(self) -> bool:
        return self.cards == tuple(range(len(self.cards)))

    def permute(self, vector: Sequence) -> Tuple:
        """Reorder a rank-indexed vector onto the new stage order."""
        vec = tuple(vector)
        if len(vec) != len(self.cards):
            raise StagePermutationIncomplete(
                f"vector of length {len(vec)} cannot be permuted onto "
                f"{len(self.cards)} stages: {list(vec)}. A rank-indexed vector "
                "that is short is not a shorter vector -- it is a vector for a "
                "different world."
            )
        return tuple(vec[c] for c in self.cards)

    def apply(self, vectors: Mapping[str, Sequence]) -> Dict[str, Tuple]:
        """Permute every rank-indexed vector, or refuse.

        The refusal is the point. ``planner/flags.py:2619`` says in so many
        words "rank 0 IS the 5090", and the KV token vector (33,13,18) and the
        reserve vector (3000,2200,2200) were measured against that sentence. A
        permutation that moves the cards but leaves those two behind hands the
        5090's KV share to a 3080 and does it QUIETLY -- the boot comes up, the
        pool is simply the wrong size on two cards. Naming the missing vector
        is cheaper than finding that at the corridor.
        """
        missing = [n for n in RANK_INDEXED_VECTORS if n not in vectors]
        if missing and not self.is_identity:
            raise StagePermutationIncomplete(
                f"cards {list(self.cards)} permutes the stages, but "
                f"{missing} were not handed over to be permuted with them. "
                "These vectors are indexed by RANK and were calibrated for the "
                "shipped 0,1,2 placement, so leaving one behind gives one card "
                "another card's budget without any error: the boot comes up and "
                "the pool is the wrong size. Pass every vector in "
                "RANK_INDEXED_VECTORS, or keep the identity permutation."
            )
        return {name: self.permute(vec) for name, vec in vectors.items()}

    def neighbour_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """The PP hops, as ordered stage pairs, under this placement."""
        return tuple((i, i + 1) for i in range(len(self.cards) - 1))

    def as_flag(self) -> str:
        """The ``--rank-gpu-id`` value this placement is."""
        return ",".join(str(c) for c in self.cards)


def slowest_hop_seconds(
    perm: StagePermutation, rates: LinkRates, nbytes: int
) -> Tuple[Tuple[int, int], float]:
    """The worst PP hop under ``perm``, and what ``nbytes`` costs on it.

    A property of the LINKS, not of a switch, and the difference caught a wrong
    conclusion while this was being written. Swapping the cards to ``0,2,1``
    does NOT halve this number: it makes the 5090's hop fast (7.13 -> 14.25
    GB/s) but leaves a 3080-to-3080 hop at 6.58 GB/s, which then becomes the
    worst one. The switch still roughly halves -- but only because the busiest
    hop CARRIES most of the bytes, not because the slowest link got faster.

    So use :func:`placement_switch_seconds` to decide a placement. This
    function answers "how bad is the worst link", which is a different and
    smaller question.
    """
    worst: Optional[Tuple[Tuple[int, int], float]] = None
    for a, b in perm.neighbour_pairs():
        for src, dst in ((a, b), (b, a)):
            secs = rates.seconds(src, dst, int(nbytes))
            if worst is None or secs > worst[1]:
                worst = ((src, dst), secs)
    if worst is None:
        raise PLayoutSwitchError("a single-stage placement has no PP hop to price.")
    return worst


def placement_switch_seconds(
    frm: PLayout, to: PLayout, layer_bytes: LayerBytes, rates: LinkRates
) -> float:
    """What the ``frm -> to`` switch actually costs under one card placement.

    The honest comparison between placements, because it prices the BYTES EACH
    HOP CARRIES rather than a notional payload on every hop. A 39->32 cut moves
    seven layers from stage 0 to stage 1 and three from stage 1 to stage 2, so
    the 5090's hop dominates and a placement that speeds it up speeds up the
    switch -- even though the pipeline's slowest LINK is unchanged.

    ``rates`` must be the table for the placement being priced: :class:`LinkRates`
    is keyed by ordered STAGE pair, and permuting the cards re-keys it.
    """
    return move_seconds(plan_layer_moves(frm, to, layer_bytes), rates)
