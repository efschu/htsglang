# SPDX-License-Identifier: Apache-2.0
"""Task #58 -- WHERE does the TRANSIENT vision tower run, and what does it cost?

WHY THIS EXISTS
---------------
User order 2026-09-20 ~10:50Z (verbatim-near): *"aktuell starten wir mit text
only und lassen den vision tower weg. im P layout soll, wenn eine vision
aufgabe kommt, ... der vision teil also bevor P layout richtig startet, auf
eine der karten geladen werden (entweder ist dort irgendwo noch vram frei,
oder wir offloaden solange einen kleinen teil in den vram), ausgefuehrt werden
und danach wieder runtergenommen werden."*

So the tower is not resident in ANY layout.  It is a stage that runs once,
before the P prefill of the request that needs it, on ONE card, and is gone
again.  This module answers the only question that has to be answered BEFORE a
single byte moves: **which card, and if none has the air, which block is
displaced for the duration -- or is this a refusal?**

It is a pure term.  No CUDA, no NVML, no file I/O, no torch: every input is a
number a boot already printed, and every output is a number a reader can check.

THE MEASUREMENTS THIS IS BUILT ON (2026-09-20, this box, desk)
--------------------------------------------------------------
Checkpoint ``/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov``,
``model.safetensors.index.json`` + the safetensors headers:

* **333 ``*visual*`` tensors, 921_460_192 bytes = 0.858 GiB, every one BF16.**
  That is the xsn63 manifest number to the byte.
* They live in ONE shard (``model-00001-of-00018.safetensors``) and they are
  **CONTIGUOUS**: first byte offset 4_889_688, last end 926_349_880, span
  926_349_880 - 4_889_688 = 921_460_192 -- exactly the sum.  The tower is a
  single sequential extent, not 333 scattered reads.  :func:`tower_from_span`
  is the constructor that says so.
* Read rate of that span, measured here: **O_DIRECT 3.85 GB/s (239 ms)**,
  buffered per-piece ``pread`` 1.08-1.15 GB/s (800-850 ms), buffered single
  span 0.93 GB/s (994 ms).  The page-cache state of the buffered runs was not
  controlled, so the honest reading is: the direct path is ~3.5x the buffered
  path on this box, and 240 ms is the floor the load has to beat.

``vision_config`` of that checkpoint: depth 27, hidden 1152, intermediate 4304,
num_heads 16, out_hidden_size 5120, patch 16, temporal_patch 2,
spatial_merge 2, **deepstack_visual_indexes = []**.  The empty deepstack list
is load-bearing for the whole design: the tower's output is a plain
``out_hidden_size``-wide tensor (5120 = the LLM's ``hidden_size``), NOT the
``out_hidden_size * (1 + len(deepstack))`` wide tensor that
``encode_server.py:441-450`` ships for deepstack checkpoints.  Nothing has to
be split, and the receiving language model needs no ``use_deepstack``.

THE LAWS THIS TERM OBEYS
------------------------
* **"Reserven NIE, nicht ein Byte"** (user 2026-09-19).  :data:`DEFAULT_FLOOR_BYTES`
  is 0 and stays 0.  A caller may pass ``floor_bytes`` with a NAMED reason; the
  plan then prints that name next to the number.
* **Transients are booked EXPLICITLY.**  The encoder's forward activation is a
  post of its own (:attr:`TowerSpec.activation_bytes`), not something the
  placement hopes is covered by slack.
* **No hand pins.**  This module chooses; it never writes a pin.  The
  placement is overridable by ``prefer_cards`` because memory
  ``vision-tower-platzierung`` records the user's 12.09. order that the
  placement be CHOOSABLE -- an override is a caller's argument here, not a
  constant in the code.
* **Refusal by name, never a silent fallback.**  Every "no" is one of the
  three exception classes below and carries the arithmetic.
* **Null only with a reached emitter.**  The encoder's *seconds* are ``None``
  unless the caller supplies a MEASURED achieved rate.  This module computes
  the FLOPs exactly (they are arithmetic from the config) and refuses to
  invent the rate that turns them into a time.

WHAT THIS TERM DOES NOT DECIDE
------------------------------
* Whether the tower CAN be built weightless and filled from that span -- that
  is slice 2 (``vision_stage_load``), and it has its own meta-device test.
* How the embeddings reach the prefill.  That path already exists upstream:
  ``MultimodalDataItem.precomputed_embeddings`` (``schedule_batch.py:800``) is
  consumed by ``_get_precomputed_embedding`` (``mm_utils.py:413-472``) BEFORE
  ``embed_mm_inputs`` would reach for ``get_image_feature``
  (``mm_utils.py:833-843``), so a P rank with ``self.visual is None``
  (``qwen3_vl.py:1286``) never dereferences the tower it does not have.
* Whether a flip is in flight.  The caller passes that fact in; this module
  only refuses on it, by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

GIB = float(1 << 30)
MIB = float(1 << 20)

#: User law 2026-09-19, verbatim "Reserven NIE, nicht ein Byte": this term
#: books no safety margin of its own.  Zero, and it stays zero.
DEFAULT_FLOOR_BYTES = 0

#: Measured on this box 2026-09-20 (see the module docstring): the O_DIRECT
#: read rate of the tower's contiguous span out of the checkpoint shard.
#: Exported so a caller does not have to retype it, NOT used as a default --
#: a rate that is not passed in is a rate that was not measured for that run.
MEASURED_ODIRECT_GBPS = 3.85
MEASURED_BUFFERED_PREAD_GBPS = 1.08


class VisionStageRefused(RuntimeError):
    """Base: the transient vision stage cannot run, and why, with numbers."""


class VisionStageNoRoom(VisionStageRefused):
    """No card has the air, and no eviction closes the gap on any card.

    Carries every card's arithmetic so the caller never has to re-derive it:
    what was free, what the stage needs, what could have been displaced, and
    by how much the best card still fell short.
    """

    def __init__(
        self,
        need_bytes: int,
        attempts: Sequence[Tuple[int, float, float, float]],
        evictions_allowed: bool,
    ):
        self.need_bytes = int(need_bytes)
        #: (card, free_bytes, evictable_bytes, shortfall_bytes) per card
        self.attempts = tuple(
            (int(c), float(f), float(e), float(s)) for c, f, e, s in attempts
        )
        self.evictions_allowed = bool(evictions_allowed)
        best = min((s for _, _, _, s in self.attempts), default=float("nan"))
        per_card = "; ".join(
            f"card{c}: free {f / GIB:.3f} GiB + evictable {e / GIB:.3f} GiB "
            f"-> short {s / GIB:.3f} GiB"
            for c, f, e, s in self.attempts
        )
        super().__init__(
            f"transient vision stage needs {self.need_bytes / GIB:.3f} GiB and no "
            f"card can hold it (evictions "
            f"{'allowed' if self.evictions_allowed else 'FORBIDDEN by the caller'}): "
            f"{per_card}. Best card is short by {best / GIB:.3f} GiB. Nothing is "
            "clamped and no reserve is raided -- the image request is refused, or "
            "a block is made evictable, or the stage waits for the next D->P flip."
        )


class VisionStageTowerUnreadable(VisionStageRefused):
    """The tower cannot be priced: its bytes are unknown or self-inconsistent.

    Raised BEFORE any placement, because a placement computed from a guessed
    weight size is a wrong answer wearing a number.
    """


class VisionStageFlipInFlight(VisionStageRefused):
    """An image request arrived while the layout was flipping.

    The stage needs a card whose free air is known and stable for its whole
    duration; during a flip the weights region is mid-move and every census
    number is stale.  The caller retries after the flip -- this is a WAIT, and
    it is named so that it never reads as a capacity failure.
    """

    def __init__(self, direction: str = "unknown"):
        self.direction = str(direction)
        super().__init__(
            f"image request reached the vision stage during a layout flip "
            f"(direction={self.direction}); the per-card census is mid-move and "
            "no placement computed from it is valid. Retry after the flip "
            "completes -- this is not a capacity refusal."
        )


@dataclass(frozen=True)
class VisionEncoderConfig:
    """The ``vision_config`` fields that determine size and FLOPs.

    Defaults are the Qwen3.8-27B-INT8-gdncov values read out of that
    checkpoint's ``config.json`` on 2026-09-20 -- they are a convenience for
    the tests, never an assumption: a caller with a different checkpoint
    passes its own.
    """

    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    out_hidden_size: int = 5120
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    #: EMPTY on this checkpoint.  Non-empty would widen the tower's output to
    #: ``out_hidden_size * (1 + len(...))`` (``encode_server.py:441-450``) and
    #: the receiving model would have to split it again
    #: (``mm_utils.py:873-877``) -- a constraint this design does not carry.
    deepstack_visual_indexes: Tuple[int, ...] = ()

    @property
    def embed_width(self) -> int:
        """Width of ONE row the tower hands to the prefill."""
        return int(self.out_hidden_size) * (1 + len(self.deepstack_visual_indexes))

    @property
    def merge_factor(self) -> int:
        return int(self.spatial_merge_size) ** 2

    def patch_rows(self, height: int, width: int, frames: int = 1) -> int:
        """Patch rows the encoder attends over for one image/clip.

        The grid is ``(t, h, w)`` in patch units; ``temporal_patch_size``
        folds frames pairwise, so a still image is one temporal patch.
        """
        if height <= 0 or width <= 0 or frames <= 0:
            raise ValueError(f"bad geometry {frames}x{height}x{width}")
        gh = height // int(self.patch_size)
        gw = width // int(self.patch_size)
        gt = max(1, math.ceil(frames / int(self.temporal_patch_size)))
        return int(gt * gh * gw)

    def vision_tokens(self, height: int, width: int, frames: int = 1) -> int:
        """Rows the PREFILL sees: patch rows after the spatial merge."""
        return int(self.patch_rows(height, width, frames) // self.merge_factor)

    def encoder_flops(self, patch_rows: int, *, full_attention: bool = True) -> int:
        """Exact multiply-accumulate FLOPs of one encoder forward.

        Arithmetic from the config, not a measurement -- which is why this
        returns FLOPs and never seconds.  Counted as 2 FLOPs per MAC.

        Per block and row: qkv ``h x 3h``, proj ``h x h``, fc1 ``h x i``,
        fc2 ``i x h``.  Attention adds ``2 x rows x rows x h`` per block when
        the tower attends fully over the image (the Qwen3-VL vision blocks
        do); a windowed tower would pass ``full_attention=False`` and get the
        linear part only, which is then a LOWER BOUND and labelled as one by
        the caller, not here.
        The merger that produces the prefill rows is counted too:
        ``(h x merge_factor) x out_hidden_size`` per OUTPUT row.
        """
        if patch_rows <= 0:
            raise ValueError(f"patch_rows={patch_rows} must be > 0")
        h = int(self.hidden_size)
        i = int(self.intermediate_size)
        per_row_per_block = 2 * (h * 3 * h + h * h + h * i + i * h)
        linear = int(patch_rows) * int(self.depth) * per_row_per_block
        attn = 0
        if full_attention:
            attn = int(self.depth) * 2 * 2 * int(patch_rows) * int(patch_rows) * h
        out_rows = int(patch_rows) // self.merge_factor
        merger = 2 * out_rows * (h * self.merge_factor) * int(self.out_hidden_size)
        return int(linear + attn + merger)


@dataclass(frozen=True)
class TowerSpec:
    """What the transient tower costs on a card, post by post.

    ``weight_bytes`` is the checkpoint truth (measured, see the module
    docstring).  ``ctx_bytes`` is the CUDA context of the process that runs
    the stage -- ZERO when the stage runs inside an existing rank process,
    non-zero when it is the 7th-rank form, and in that case it is a number the
    caller MEASURED, because a context size guessed from a driver version is
    exactly the kind of number that later eats a prefill.
    ``activation_bytes`` is the encoder forward's transient: booked
    explicitly, per the user law, never left to slack.
    """

    pieces: int
    weight_bytes: int
    ctx_bytes: int = 0
    activation_bytes: int = 0
    #: bytes of the embeddings the stage hands on.  They leave the card (to
    #: host-pinned or straight to the P stage-0 buffer), but they exist ON the
    #: card while the encoder writes them, so they are a post.
    embedding_bytes: int = 0
    #: purely for the report: the checkpoint extent this came from.
    source: str = ""

    def __post_init__(self):
        if self.pieces <= 0:
            raise VisionStageTowerUnreadable(
                f"tower manifest lists {self.pieces} pieces; a tower with no "
                "pieces cannot be placed (and would silently place anywhere)"
            )
        if self.weight_bytes <= 0:
            raise VisionStageTowerUnreadable(
                f"tower weight_bytes={self.weight_bytes}; refusing to place a "
                "tower whose size was not read from the checkpoint"
            )
        for name in ("ctx_bytes", "activation_bytes", "embedding_bytes"):
            if getattr(self, name) < 0:
                raise VisionStageTowerUnreadable(f"{name} must be >= 0")

    @property
    def total_bytes(self) -> int:
        """Everything the stage occupies on the chosen card at its peak."""
        return int(
            self.weight_bytes
            + self.ctx_bytes
            + self.activation_bytes
            + self.embedding_bytes
        )

    @property
    def posts(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("tower weights", int(self.weight_bytes)),
            ("cuda context", int(self.ctx_bytes)),
            ("encoder activation", int(self.activation_bytes)),
            ("embeddings", int(self.embedding_bytes)),
        )


def tower_from_span(
    pieces: int,
    first_offset: int,
    last_end: int,
    byte_sum: int,
    *,
    shard: str = "",
    **kw,
) -> TowerSpec:
    """Build a :class:`TowerSpec` from a checkpoint extent, and CHECK it.

    The check is the point: ``last_end - first_offset == byte_sum`` is what
    makes the load a single sequential read instead of ``pieces`` seeks.  When
    it does not hold the tower is scattered -- that is not an error, but the
    caller must not then plan on the contiguous read rate, so this refuses and
    says so rather than handing back a spec that reads as contiguous.
    """
    span = int(last_end) - int(first_offset)
    if span != int(byte_sum):
        raise VisionStageTowerUnreadable(
            f"tower is NOT one contiguous extent in {shard or '<shard>'}: span "
            f"{span} bytes ({first_offset}..{last_end}) against a manifest sum of "
            f"{byte_sum} bytes over {pieces} pieces -- a difference of "
            f"{span - int(byte_sum)} bytes. Plan the load as {pieces} reads, not "
            "as one; this constructor refuses to imply otherwise."
        )
    return TowerSpec(
        pieces=int(pieces),
        weight_bytes=int(byte_sum),
        source=f"{shard}[{first_offset}:{last_end}]" if shard else "",
        **kw,
    )


@dataclass(frozen=True)
class EvictableBlock:
    """A weights block that may be displaced to host RAM for the stage.

    ``name`` is the memory-saver tag / module prefix that names it -- the
    refusal and the plan both print it, so an operator reading the log knows
    exactly what moved.  ``bytes`` is what it frees on the card.  The two
    rates are the MEASURED link rates for the two directions; they are not
    assumed equal, because on this rig they are not (memory
    RANG-LINK-ZUORDNUNG: rank 1 is x4, ranks 0 and 2 are x8).
    """

    name: str
    bytes: int
    out_gbps: float
    in_gbps: float
    #: True when the block can be displaced WITHOUT invalidating a captured
    #: CUDA graph.  A block inside a graph's private pool is not evictable at
    #: all -- it is listed with ``graph_safe=False`` so the refusal can name
    #: it as present-but-untouchable instead of pretending it is absent.
    graph_safe: bool = True

    def __post_init__(self):
        if self.bytes <= 0:
            raise ValueError(f"evictable block {self.name!r}: bytes must be > 0")
        if self.out_gbps <= 0 or self.in_gbps <= 0:
            raise ValueError(
                f"evictable block {self.name!r}: link rates must be > 0 "
                "(pass the measured rate, never a placeholder)"
            )

    @property
    def evict_seconds(self) -> float:
        return float(self.bytes) / (float(self.out_gbps) * 1e9)

    @property
    def restore_seconds(self) -> float:
        return float(self.bytes) / (float(self.in_gbps) * 1e9)

    @property
    def round_trip_seconds(self) -> float:
        return self.evict_seconds + self.restore_seconds


@dataclass(frozen=True)
class CardAir:
    """One card's free air in the P layout, as a previous boot measured it.

    ``free_bytes`` is the **IDLE** free air -- ``free_idle_mib`` in
    ``weg2/corridor_budget.py:143`` terms, the ``[vram-peak] ... card free X
    of Y GiB`` reading taken with no prefill in flight.  That is the right
    instrument for THIS stage and the load reading is not, because the user
    order puts the stage *"bevor P layout richtig startet"*: the tower is up
    and gone again before the first prefill chunk draws its transient, so the
    two never coexist and the stage must not be sized against a number that
    already has the prefill subtracted.

    ``free_under_load_bytes`` is the load reading (``free_load_mib``) and is
    carried for the REPORT only: it is what the plan prints to show a reader
    that the stage is not eating the prefill's transient, since by then it is
    gone.  It never enters the placement arithmetic.  The one real
    sequencing duty it implies is named in the plan: any displaced block must
    be BACK before the prefill starts.

    ``h2d_gbps`` is this card's MEASURED host-to-device rate, not its nominal
    link width (on this rig the two differ by a factor of two between the x4
    and the x8 slots -- memory RANG-LINK-ZUORDNUNG).
    """

    card: int
    ranks: Tuple[int, ...]
    total_bytes: int
    free_bytes: int
    h2d_gbps: float
    evictable: Tuple[EvictableBlock, ...] = ()
    #: report-only, never placed against.  See the class docstring.
    free_under_load_bytes: Optional[int] = None
    #: free-form, printed in the plan: e.g. "fn8aj [vram-peak] extend".
    provenance: str = ""

    def __post_init__(self):
        if self.free_bytes < 0:
            raise ValueError(f"card{self.card}: free_bytes must be >= 0")
        if self.total_bytes <= 0:
            raise ValueError(f"card{self.card}: total_bytes must be > 0")
        if self.free_bytes > self.total_bytes:
            raise ValueError(
                f"card{self.card}: census is self-inconsistent -- free "
                f"{self.free_bytes} exceeds total {self.total_bytes}"
            )
        if self.h2d_gbps <= 0:
            raise ValueError(
                f"card{self.card}: h2d_gbps must be > 0 (measured rate, not a "
                "nominal link width)"
            )

    @property
    def evictable_bytes(self) -> int:
        """Only what is ACTUALLY displaceable: a block inside a captured
        graph's private pool is not."""
        return int(sum(b.bytes for b in self.evictable if b.graph_safe))


@dataclass(frozen=True)
class VisionStagePlan:
    """The answer: this card, these blocks displaced, these seconds."""

    card: int
    ranks: Tuple[int, ...]
    tower: TowerSpec
    free_before_bytes: int
    need_bytes: int
    evicted: Tuple[EvictableBlock, ...]
    floor_bytes: int
    floor_reason: str
    #: free air on the card once the stage is fully up, before it tears down.
    slack_bytes: float
    #: seconds, each a named leg.  ``encode`` is ``None`` unless a MEASURED
    #: achieved FLOP rate was supplied -- FLOPs are arithmetic, seconds are not.
    read_seconds: float
    h2d_seconds: float
    load_seconds: float
    evict_seconds: float
    restore_seconds: float
    encode_flops: int
    encode_seconds: Optional[float]
    #: report-only (see :class:`CardAir`): the card's free air WITH a prefill
    #: in flight.  Printed so a reader sees that the stage does not coexist
    #: with the prefill transient; never placed against.
    free_under_load_bytes: Optional[int] = None
    #: candidates that were considered and why they lost, for the report.
    rejected: Tuple[Tuple[int, str], ...] = ()

    @property
    def stage_seconds(self) -> Optional[float]:
        """Wall time from "image arrives" to "tower is gone again", as far as
        it can be priced.  ``None`` when the encode leg was not measurable --
        a partial sum printed as a total is the denominator trap."""
        if self.encode_seconds is None:
            return None
        return (
            self.evict_seconds
            + self.load_seconds
            + self.encode_seconds
            + self.restore_seconds
        )

    def report(self) -> str:
        lines = [
            f"vision stage -> card{self.card} (ranks {list(self.ranks)}), "
            f"need {self.need_bytes / GIB:.3f} GiB of "
            f"{self.free_before_bytes / GIB:.3f} GiB free, "
            f"slack after {self.slack_bytes / GIB:.3f} GiB",
        ]
        for name, val in self.tower.posts:
            lines.append(f"  post {name:<20} {val / MIB:9.1f} MiB")
        if self.floor_bytes:
            lines.append(
                f"  post {'caller floor':<20} {self.floor_bytes / MIB:9.1f} MiB "
                f"({self.floor_reason or 'UNNAMED -- the law says name it'})"
            )
        for b in self.evicted:
            lines.append(
                f"  displaced {b.name!r}: {b.bytes / MIB:.1f} MiB "
                f"out {b.evict_seconds * 1e3:.0f} ms / back {b.restore_seconds * 1e3:.0f} ms"
            )
        lines.append(
            f"  legs (ms): evict {self.evict_seconds * 1e3:.0f}, "
            f"read {self.read_seconds * 1e3:.0f}, h2d {self.h2d_seconds * 1e3:.0f} "
            f"-> load {self.load_seconds * 1e3:.0f}, "
            f"encode {'n/a (rate not measured)' if self.encode_seconds is None else f'{self.encode_seconds * 1e3:.0f}'}, "
            f"restore {self.restore_seconds * 1e3:.0f}"
        )
        lines.append(f"  encode work: {self.encode_flops / 1e9:.1f} GFLOP")
        if self.free_under_load_bytes is not None:
            lines.append(
                f"  (card free WITH a prefill in flight: "
                f"{self.free_under_load_bytes / GIB:.3f} GiB -- the stage is gone "
                "by then; any displaced block must be BACK before the prefill)"
            )
        for card, why in self.rejected:
            lines.append(f"  card{card} rejected: {why}")
        return "\n".join(lines)


def _minimal_eviction(
    blocks: Sequence[EvictableBlock], gap_bytes: float
) -> Optional[Tuple[EvictableBlock, ...]]:
    """Smallest displacement that closes ``gap_bytes``.

    "Smallest" means fewest bytes moved, because bytes moved is what the user
    order calls "ein kleiner Teil" and what the round trip is paid in.  Greedy
    over blocks sorted by size ascending, taking the first block that closes
    the gap alone if one does -- so a 4 GiB gap does not displace a 16 GiB
    block when a 5 GiB one is on the list.  Returns ``None`` when the whole
    list is not enough.
    """
    usable = sorted(
        (b for b in blocks if b.graph_safe), key=lambda b: (b.bytes, b.name)
    )
    if gap_bytes <= 0:
        return ()
    single = [b for b in usable if b.bytes >= gap_bytes]
    if single:
        return (single[0],)
    # No single block is enough: accumulate from the largest down, which moves
    # the fewest blocks, then drop any block the remainder no longer needs.
    chosen: list[EvictableBlock] = []
    got = 0.0
    for b in sorted(usable, key=lambda b: (-b.bytes, b.name)):
        chosen.append(b)
        got += b.bytes
        if got >= gap_bytes:
            break
    if got < gap_bytes:
        return None
    for b in sorted(chosen, key=lambda b: b.bytes):
        if got - b.bytes >= gap_bytes:
            chosen.remove(b)
            got -= b.bytes
    return tuple(sorted(chosen, key=lambda b: b.name))


def plan_vision_stage(
    cards: Iterable[CardAir],
    tower: TowerSpec,
    *,
    read_gbps: float,
    prefer_cards: Sequence[int] = (),
    allow_eviction: bool = True,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    floor_reason: str = "",
    encoder_flops: int = 0,
    achieved_tflops: Optional[float] = None,
    pipelined_load: bool = True,
    flip_in_flight: bool = False,
    flip_direction: str = "unknown",
) -> VisionStagePlan:
    """Choose the card the transient tower runs on -- or refuse, by name.

    Order of decision, and each step is a rule a reader can check:

    1. ``flip_in_flight`` -> :class:`VisionStageFlipInFlight`.  A census taken
       mid-flip is stale and every number below would be fiction.
    2. Cards that fit with NO eviction win over cards that need one.  Moving
       zero bytes is always cheaper than moving some.
    3. Within a group, ``prefer_cards`` order wins if the caller gave one
       (memory ``vision-tower-platzierung``: the placement is CHOOSABLE).
       Otherwise the tie-break is, in order: lowest total stage seconds
       (which on this rig is the x8/x4 link difference), then lowest card
       index.  It is deliberately NOT "the biggest card": the 12.09. user
       order says the 5090's space is the most valuable on the rig, and a
       link-time tie-break lets a 3080 on x8 win over the 5090 whenever it
       can, without hardcoding either.
    4. Nothing fits -> :class:`VisionStageNoRoom` with every card's
       arithmetic.

    ``read_gbps`` is the MEASURED checkpoint read rate (see
    :data:`MEASURED_ODIRECT_GBPS`).  ``achieved_tflops`` is the MEASURED
    encoder rate; without it ``encode_seconds`` is ``None`` and
    ``stage_seconds`` is ``None`` too, rather than a partial sum wearing the
    name of a total.
    """
    if flip_in_flight:
        raise VisionStageFlipInFlight(flip_direction)
    if read_gbps <= 0:
        raise ValueError("read_gbps must be > 0 (the measured rate, not a guess)")
    if floor_bytes < 0:
        raise ValueError("floor_bytes must be >= 0")
    if floor_bytes and not floor_reason:
        raise ValueError(
            "floor_bytes without floor_reason: the user law is 'Reserven NIE, "
            "nicht ein Byte'; a non-zero floor is allowed only WITH a named "
            "reason, which the plan then prints"
        )

    need = tower.total_bytes + int(floor_bytes)
    cards = list(cards)
    if not cards:
        raise VisionStageNoRoom(need, (), allow_eviction)

    prefer = {c: i for i, c in enumerate(prefer_cards)}
    read_seconds = float(tower.weight_bytes) / (float(read_gbps) * 1e9)
    encode_seconds = (
        None
        if achieved_tflops is None
        else float(encoder_flops) / (float(achieved_tflops) * 1e12)
    )

    attempts: list[Tuple[int, float, float, float]] = []
    candidates: list[Tuple[Tuple[int, int, float, int], VisionStagePlan]] = []

    for c in cards:
        gap = need - float(c.free_bytes)
        evicted: Tuple[EvictableBlock, ...] = ()
        if gap > 0:
            if not allow_eviction:
                attempts.append((c.card, float(c.free_bytes), 0.0, gap))
                continue
            picked = _minimal_eviction(c.evictable, gap)
            if picked is None:
                attempts.append(
                    (
                        c.card,
                        float(c.free_bytes),
                        float(c.evictable_bytes),
                        gap - float(c.evictable_bytes),
                    )
                )
                continue
            evicted = picked
        attempts.append((c.card, float(c.free_bytes), float(c.evictable_bytes), 0.0))

        h2d_seconds = float(tower.weight_bytes) / (float(c.h2d_gbps) * 1e9)
        load_seconds = (
            max(read_seconds, h2d_seconds)
            if pipelined_load
            else read_seconds + h2d_seconds
        )
        evict_seconds = sum(b.evict_seconds for b in evicted)
        restore_seconds = sum(b.restore_seconds for b in evicted)
        freed = sum(b.bytes for b in evicted)
        plan = VisionStagePlan(
            card=c.card,
            ranks=tuple(c.ranks),
            tower=tower,
            free_before_bytes=int(c.free_bytes),
            need_bytes=int(need),
            evicted=evicted,
            floor_bytes=int(floor_bytes),
            floor_reason=str(floor_reason),
            slack_bytes=float(c.free_bytes) + freed - need,
            read_seconds=read_seconds,
            h2d_seconds=h2d_seconds,
            load_seconds=load_seconds,
            evict_seconds=evict_seconds,
            restore_seconds=restore_seconds,
            encode_flops=int(encoder_flops),
            encode_seconds=encode_seconds,
            free_under_load_bytes=c.free_under_load_bytes,
        )
        movement_seconds = evict_seconds + load_seconds + restore_seconds
        # The tie-break on the link is NOT redundant with movement_seconds,
        # and the reason is measured: with the pipelined load the two legs are
        # max(read, h2d), and on this box the O_DIRECT read is 3.85 GB/s while
        # every card's H2D is faster (6.5 GB/s on the x4, 13.3-14.4 on the
        # x8) -- so the READ dominates and the modelled times TIE across
        # cards.  Falling through to the card index there would pick by
        # enumeration order, which is not a reason.  The link is: it is the
        # leg whose rate the pipelining assumption is least sure about, so on
        # a tie the faster link wins.
        key = (
            1 if evicted else 0,
            prefer.get(c.card, len(prefer)),
            movement_seconds,
            -float(c.h2d_gbps),
            c.card,
        )
        candidates.append((key, plan))

    if not candidates:
        raise VisionStageNoRoom(need, attempts, allow_eviction)

    candidates.sort(key=lambda kp: kp[0])
    _, best = candidates[0]
    rejected = []
    for key, plan in candidates[1:]:
        delta_ms = (key[2] - candidates[0][0][2]) * 1e3
        if plan.evicted:
            why = f"would displace {sum(b.bytes for b in plan.evicted) / MIB:.0f} MiB"
        elif delta_ms > 0.5:
            why = f"slower by {delta_ms:.0f} ms"
        else:
            why = (
                f"same modelled time (the read leg dominates), slower link "
                f"{-key[3]:.1f} vs {-candidates[0][0][3]:.1f} GB/s"
            )
        rejected.append((plan.card, why))
    for card, free_b, evict_b, short in attempts:
        if short > 0:
            rejected.append(
                (card, f"short by {short / GIB:.3f} GiB (free {free_b / GIB:.3f} GiB)")
            )
    return VisionStagePlan(
        card=best.card,
        ranks=best.ranks,
        tower=best.tower,
        free_before_bytes=best.free_before_bytes,
        need_bytes=best.need_bytes,
        evicted=best.evicted,
        floor_bytes=best.floor_bytes,
        floor_reason=best.floor_reason,
        slack_bytes=best.slack_bytes,
        read_seconds=best.read_seconds,
        h2d_seconds=best.h2d_seconds,
        load_seconds=best.load_seconds,
        evict_seconds=best.evict_seconds,
        restore_seconds=best.restore_seconds,
        encode_flops=best.encode_flops,
        encode_seconds=best.encode_seconds,
        free_under_load_bytes=best.free_under_load_bytes,
        rejected=tuple(rejected),
    )
