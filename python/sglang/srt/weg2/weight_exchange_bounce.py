# SPDX-License-Identifier: Apache-2.0
"""Weg-2 #1273 S6-BOUNCE -- the AUTHORITATIVE weight bytes, with no dormant
image in system RAM.

Design of record: ``/spinning/gpu-arb/weg2/WEG2_REUSE_SPEC_0908.md`` section 10.
Plan of record: ``/spinning/gpu-arb/weg2/PLAN_S6_BOUNCE_0911.md`` steps B2/B3.
User law (verbatim in ``memory/gewichtsaustausch-ziel-kein-dauer-hostram.md``):
*"notfalls wird das layer auf einem (vertretbar kleinen) hostpuffer
vollstaendig zusammengesetzt und jede karte nimmt sich von dem was er braucht
(oder ihn komplett). die 27gb (oder so aehnlich) an layerbytes muessen nicht
mehr dauerhaft im systemram gehalten werden. das ist das ziel."*

WHAT THIS REPLACES, and why the replacement is not an optimisation.  Under
``--weg2-weight-source ring`` the dormant weight image lives in a preallocated
host region at Sigma H and the legs copy THROUGH it, so it is charged once and
never shrinks for the life of the boot (measured on boot weg2xsn8: ARM line
``host weights term 42.96 GiB``).  Under ``exchange`` the weights region is
opened with ``enable_cpu_backup=False``, so ``pause`` is a pure unmap and
``resume`` a pure remap of pages whose CONTENT IS UNDEFINED -- and the only
thing that refilled them was ``update_weights_from_disk``
(``weight_updater.py:743``, measured 12.073/14.143/16.749 s per wake on this
rig).  This module is what puts the bytes there instead, out of the peer
group's live VRAM, through a buffer whose size is bounded by construction.

TWO PATHS, because P and D do not agree about a tensor's shape
--------------------------------------------------------------
* **path (a), byte-identical pieces** -- DELETED #1342 S3, and named here so
  the removal is not mistaken for an omission.  Where the source's piece and
  the destination's piece were the same bytes, a separate lane moved them
  card-to-card.  It had no production caller, its input (the per-flip-leg
  agreement verdict from ``reconcile_card_manifest``) does not exist at the one
  call site that reaches this module, and path (b) carries those bytes anyway.
  Measured on boot weg2xsn8 the set was 4.90 MiB against a 27.52 GiB image --
  0.018 %, so it was a SECOND MOVER for one payload rather than a throughput
  argument.  ``agreed_descs``, ``AgreedResult`` and ``run_agreed_leg`` went
  with it.
* **path (b), everything else** -- P is PP (whole layers per stage) and D is TP
  (row-sliced), so there is no byte-identical piece to move.  The source
  deposits the unit into the host bounce buffer and each destination collects
  ITS OWN ROWS.  This carries the other ~99.98 %.

SIZING: AMENDMENT 3 SUPERSEDES SECTION 10.2, WHICH WAS A MEAN
-------------------------------------------------------------
Section 10.2 sized the assemble buffer at ``bytes_per_direction / n_layers``
(27.12 GiB / 64 = 433.9 MiB) while section 10.5 bounded the refusal at the
WIDEST layer, "not the mean; sizing on the mean is how a boot dies on layer
47".  Those two cannot both hold, and the checkpoint census settled it
(PLAN_S6_BOUNCE_0911 AMENDMENT 3, from ``weg2/checkpoint_census.py``;
independently reproduced by ``/spinning/gpu-arb/weg2/tools/layer_unit_census.py``
-- both read the same 18 safetensors headers, no GPU, and agree on the widest
layer to the BYTE):

    WIDEST layer  = layers.0 = 756,323,776 B = 721 MiB   (1.66x the mean)
    mean layer    = 386.9 MiB                (this module never computes with it)
    layered total = 24.76 GiB, unlayered = 4.79 GiB (embed / lm_head / MTP)
    widest UNLAYERED class = lm_head.weight = 2425 MiB
    widest indivisible ROW = 17408 B

Layer 0 is the widest because it carries the linear-attention family
(``in_proj_qkv``/``z``/``a``/``b``, ``conv1d``, ``A_log``, ``dt_bias``) ON TOP
OF the MLP triple -- so the widest layer is a DIFFERENT SHAPE from the mean
layer, not merely a bigger one, and a double that models only the MLP triple
would be testing the wrong form.

Buffer = ``widest x depth 2`` = 1442 MiB, path (a) staging 384 MiB, bounce
total 1826 MiB = 1.78 GiB, released 42.96 -> 1.78 GiB (24x).  A section-10.2
buffer (868 MiB) could not have assembled the widest layer at all.

THE DECIDE THIS MODULE MAKES: A LAYER IS ASSEMBLED WHOLE, AN UNLAYERED CLASS
IS BANDED (2026-09-11, appended to section 10, reported to the operator).
``lm_head`` at 2425 MiB does not fit the 721 MiB slot, so the two options are
a slot sized on ``max(layer, unlayered)`` -- 4850 MiB buffer, 5.11 GiB bounce
total, 37.85 GiB released (8.4x) -- or the widest LAYER with the unlayered
classes cut into row bands: 1443 MiB, 1.78 GiB, 41.18 GiB released (24.1x).
The second IS AMENDMENT 3's own 1826 MiB / 24x, so the plan's figure already
presupposes it; the first would triple the host residency the law exists to
shrink, for two tensors.  And the law's word is *"das layer"*: ``lm_head`` is
not a layer.  ``lm_head`` takes 4 bands at a 721 MiB slot, a band is a
complete set of ROWS, and each destination's slice of a band is the
descriptor's own row map -- so banding costs correctness nothing.  What it
costs is that the class is not resident whole, which the law does not ask for
outside a layer, and :attr:`BounceResult.banded` prints which classes those
were so it is answerable from the log rather than argued.

A LAYER above the slot is therefore still a REFUSAL
(:func:`refuse_if_plan_exceeds_slot`), and the band mechanism is not an escape
hatch from it.  The cut itself is not a new mechanism either: it is
``transport.batch_descs``, which already "splits a STRIDED2D descriptor by
ROWS, a row being the smallest unit whose pitch arithmetic stays exact", and
already refuses (W68) a run above the slot -- the floor no cut can go below,
measured at 17408 B and therefore unable to fire at any sane slot.

NO NEW W-CODE (section 10.5's closing sentence).  The refusals reuse
``Weg2XchgPlanDisagree`` (W68) for a run the slot cannot hold and
``Weg2XchgSourceMissing`` (W74) for a destination whose slice no source
covers.  Assembly does not create a source; it only stages one.

DEADLOCK IS A REFUSAL, NOT A HANG.  A depth-``d`` buffer is only correct while
the destination is draining what the source deposits: the sleep and wake legs
are concurrent on co-located ranks (C9, see ``weight_updater.py``'s C14 credit
wait), and if the drain stops the source blocks holding bytes it cannot place.
That is the same condition the cross lane already bounds with a named W69
timeout, and the store-and-forward carrier here keeps the same contract: the
FILE outlives the leg that wrote it (:class:`LayerBounce` does not unlink), so
a source may deposit, close its own mapping and return while the destination
opens the same path in a LATER leg and finds the bytes.
"""

from __future__ import annotations

import ctypes
import logging
import mmap as _mmap
import os
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.mem_cache.pinned_host_budget import (
    check_and_register_pinned_post,
    registered_posts,
    revert_pinned_posts_on_failure,
    unregister_pinned_post,
)
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp
from sglang.srt.weg2 import xchg_bounce as xb

logger = logging.getLogger(__name__)

MIB = xr.MIB

__all__ = [
    "BOUNCE_POST_FLAG",
    "BounceResult",
    "LayerBounce",
    "Unit",
    "bounce_path",
    "leg_geometry",
    "plan_units",
    "refuse_if_plan_exceeds_slot",
    "refuse_if_slot_short",
    "registered_bounce_bytes",
    "InjectVerdict",
    "inject_summary_line",
    "reset_inject_verdicts",
    "run_bounce_leg",
    "unit_name",
    "widest_run",
    "widest_unit",
]

#: The flag an operator lowers to shrink this buffer.  Named because a refusal
#: that cannot be acted on is half a refusal -- the same reason
#: :class:`tp.HostBounce` carries ``POST_FLAG``.
BOUNCE_POST_FLAG = "--weg2-xchg-bounce-slot-mib"

#: ``model.language_model.layers.7.mlp.down_proj.weight`` -> ``layers.7``.
#: The LAYER is the unit because that is what section 10.3 assembles and what
#: the wave gate already bounds; a parameter with no layer index (``lm_head``,
#: ``embed_tokens``, the vision tower) is its own unit, which is exactly how
#: the census counted the 342 non-layer units above.
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def unit_name(param_name: str) -> str:
    """The streaming unit a parameter belongs to."""
    m = _LAYER_RE.search(str(param_name))
    return f"layers.{int(m.group(1))}" if m else str(param_name)


@dataclass(frozen=True)
class Unit:
    """One streaming unit: a layer group-wide, or one non-layer parameter.

    ``descs`` is EVERY descriptor of that unit across EVERY destination -- the
    "assembled completely" of the user law -- and ``nbytes`` is their sum, so
    a unit's cost is the group-wide cost and not one card's share.
    """

    key: Tuple[str, str]
    descs: Tuple[object, ...]
    nbytes: int

    @property
    def tag(self) -> str:
        return self.key[0]

    @property
    def is_layer(self) -> bool:
        """Is this unit a decoder LAYER, or an unlayered class?

        The distinction decides whether the unit must be assembled COMPLETE in
        one depth-slot.  The user law says *"das layer ... vollstaendig
        zusammengesetzt"* -- it speaks of a LAYER, and ``lm_head`` (2425 MiB
        measured) is not one.  See :func:`refuse_if_plan_exceeds_slot` for the
        decide this property carries.
        """
        return self.key[1].startswith("layers.")

    def bands(self, slot_bytes: int) -> int:
        """How many slot-sized bands this unit is cut into."""
        return len(tp.batch_descs(list(self.descs), int(slot_bytes)))


def plan_units(descs: Sequence[object]) -> List[Unit]:
    """Group descriptors into streaming units, IN PLAN ORDER.

    First-appearance order, not sorted: the plan's own order is what both ends
    derive independently (``batch_descs``' determinism argument), and a sort
    here would be a second ordering authority for the same bytes.  ZEROFILL
    descriptors have no source and are excluded -- the destination memsets
    them locally (``tp.apply_zerofill``), so staging them would move padding
    as payload.
    """
    order: List[Tuple[str, str]] = []
    grouped: Dict[Tuple[str, str], List[object]] = {}
    for d in descs:
        if getattr(d, "kind", None) == wx.ZEROFILL:
            continue
        key = (str(getattr(d, "tag", "")), unit_name(getattr(d, "param_name", "")))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(d)
    return [
        Unit(key=key, descs=tuple(grouped[key]),
             nbytes=sum(int(getattr(d, "nbytes", 0)) for d in grouped[key]))
        for key in order
    ]


def widest_unit(units: Sequence[Unit]) -> Unit:
    """The widest unit BY NAME, which is what section 10.5 asks to be reported.

    ``max`` over bytes with the KEY as the tie-break, so the answer is stable
    across processes: two units of equal width must not be reported
    differently by the two ends of the same leg.
    """
    if not units:
        raise ValueError("no units -- a plan with no source is W74's case")
    return max(units, key=lambda u: (u.nbytes, u.key))


def widest_run(descs: Sequence[object]) -> int:
    """The widest INDIVISIBLE run in bytes -- the hard floor for any slot.

    A 2-D copy may be cut at a row boundary but never inside a row, so this is
    the one quantity below which no slot size can be made to work.  FLAT
    descriptors are cut at arbitrary byte boundaries and therefore impose no
    floor; only STRIDED2D runs do.
    """
    runs = [
        int(getattr(d, "run_bytes", 0))
        for d in descs
        if getattr(d, "kind", None) == wx.STRIDED2D
    ]
    return max(runs) if runs else 0


def refuse_if_slot_short(slot_bytes: int, descs: Sequence[object]) -> None:
    """Refuse a slot no cut of this plan can survive.  W68, no new code.

    THE BOUND IS THE WIDEST RUN AND NOT THE WIDEST UNIT, and the difference is
    the whole stated deviation of this module: under the band cut a slot
    smaller than a unit is correct, so refusing on the unit would refuse
    working boots, while refusing on the mean -- which is what section 10.2's
    433.9 MiB is -- would accept a slot the widest run cannot fit.  Delegated
    to ``tp.batch_descs`` rather than re-deriving the comparison, so there is
    ONE producer of this verdict and the message is the transport's own.
    """
    if int(slot_bytes) <= 0:
        raise ValueError(f"slot_bytes must be positive, not {slot_bytes!r}")
    # The batcher raises W68 on the first run it cannot place.  Calling it for
    # its refusal is deliberate: a second comparison here could disagree with
    # the one that actually stages, which is the two-ledgers defect.
    tp.batch_descs(list(descs), int(slot_bytes))


def refuse_if_plan_exceeds_slot(slot_bytes: int, descs: Sequence[object],
                                terms: Optional[xb.BounceTerms] = None) -> None:
    """Refuse when THIS PLAN's widest unit does not fit one depth-slot.

    THE SAME REFUSAL AS ARM TIME, AT THE OTHER MOMENT, and that is the point
    rather than a duplicate.  ``xchg_bounce.bounce_terms`` grades a NUMBER --
    ``widest_layer_bytes``, measured by the launch-time checkpoint census
    (step B1b) -- while this grades the PLAN the leg actually holds.  If the
    census under-read the widest layer, arm time says ``covers_widest=yes`` and
    the assembly would then band a unit the user law requires assembled
    COMPLETE (AMENDMENT 2).  That gap is invisible to either check alone.

    It raises ``Weg2XchgBounceUnderCovered`` with
    ``xchg_bounce.under_coverage_refusal``'s own text -- one producer of the
    message, so the log cannot carry two different spellings of one refusal.
    """
    units = plan_units(descs)
    if not units:
        return
    # THE DECIDE (AMENDMENT 3 follow-up, 2026-09-11): only a LAYER must be
    # assembled complete in one slot.  An UNLAYERED class is banded instead.
    #
    # Forced by the plan's own arithmetic rather than chosen: measured on this
    # checkpoint's headers the widest layer is 756,323,776 B = 721 MiB while
    # `lm_head.weight` is 2425 MiB and `embed_tokens` 1212 MiB.  Sizing the
    # slot on `max(layer, unlayered)` costs a 4850 MiB buffer, a 5.11 GiB
    # bounce total and releases 37.85 GiB (8.4x); sizing it on the widest LAYER
    # and banding the unlayered classes costs 1443 MiB, a 1.78 GiB total and
    # releases 41.18 GiB (24.1x) -- which IS AMENDMENT 3's own 1826 MiB / 24x.
    # So the amendment's figure already presupposes this decide, and option A
    # would triple the host residency the law exists to shrink, for two
    # tensors, while the law's word is "layer".
    #
    # `lm_head` takes 4 bands at a 721 MiB slot.  A band is a complete set of
    # ROWS and each destination's slice of it is the descriptor's own row map,
    # so banding costs correctness nothing; what it costs is that the class is
    # not resident whole, which the law does not ask for outside a layer.
    layered = [u for u in units if u.is_layer]
    widest = widest_unit(layered) if layered else None
    if widest is None or widest.nbytes <= int(slot_bytes):
        return
    if terms is None:
        # No ARM term to quote (a caller that sized by hand, i.e. a test or a
        # tool).  The refusal still names the same two numbers rather than
        # inventing a second message shape.
        terms = xb.bounce_terms(
            bytes_per_direction=sum(u.nbytes for u in units),
            n_layers=len(units), widest_layer_bytes=widest.nbytes,
            pairs=0, depth=1, slot_bytes=xr.SLOT_BYTES,
        )
    raise xb.Weg2XchgBounceUnderCovered(
        xb.under_coverage_refusal(terms, widest_layer_name=widest.key[1])
        + f" MEASURED AT RUN TIME from the plan itself: the widest unit is "
          f"{widest.key[1]} at {widest.nbytes} B against a depth-slot of "
          f"{int(slot_bytes)} B, so the launch-time census that sized the "
          f"buffer under-read this boot's widest layer."
    )


def leg_geometry(terms: xb.BounceTerms) -> Tuple[int, int]:
    """``(slot_bytes, depth)`` for :func:`run_bounce_leg`, out of the ARM's own
    term -- THE ONE READER of that decision.

    Section 10.8's instruction for step 3, applied to step 4 as well: the
    region layout and ``bounce_terms(slot_bytes=)`` "must be fed the SAME
    value -- one reader, or they drift".  So this module derives NO size of its
    own.  ``xchg_bounce.bounce_terms`` sets ``buffer_bytes =
    widest_layer_bytes * depth``, which makes the per-slot width the WIDEST
    LAYER and is why a unit is assembled COMPLETE in one slot (AMENDMENT 2,
    the user law's "vollstaendig zusammengesetzt").
    """
    # THE WIDEST LAYER, NAMED -- not `buffer_bytes // depth`, which was a THIRD
    # expression for the same width and silently assumed
    # `buffer = widest * depth`. It stopped being true the moment the price
    # started counting the shadow's extra slot (xchg_bounce.assemble_slots),
    # and it would have returned `widest * 3 // 2` -- a per-slot width no slot
    # has. The term already carries the number under its own name.
    return int(terms.widest_layer_bytes), int(terms.depth)


def leg_slot_bytes(terms: xb.BounceTerms) -> int:
    """The size of ONE band :func:`run_bounce_leg` actually allocates --
    THE ONE PRODUCER ``LayerBounce``'s ``slot_bytes=`` argument is built from.

    #1385/xsn31-4 ROOT CAUSE THIS FUNCTION FIXES. Before it, ``run_bounce_leg``
    read :func:`leg_geometry` for ``slot_bytes`` UNCONDITIONALLY -- the
    PRE-#1374 AMENDMENT-2 geometry, where one band equals one WHOLE LAYER --
    while separately reading ``terms.lane_slots`` (Option 1's band COUNT,
    computed by :func:`xchg_bounce.tag_slots` under the assumption that a
    band is ``terms.slot_bytes`` wide, the 128 MiB oncard unit) for the band
    count. Multiplying a count sized for ``terms.slot_bytes``-wide bands by
    the WIDEST-LAYER size instead inflated the real buffer past what the ARM
    priced: MEASURED on boot weg2xsn31/4 (two independent instruments,
    deckungsgleich -- a 0.2 s file-system poller and the lane's own
    WEG2-XCHG-HOST-SLOT lines), the allocated file was
    ``756,323,776 B x 24 = 18,151,770,624 B`` (16.905 GiB) against a ledger
    charge of ``134,217,728 B x 24 = 3,221,225,472 B`` (3.00 GiB) for that
    same lane -- 5.6x over, with only 2 of 5 lanes active (cushion 0.19 GiB
    against the 1.50 GiB floor, 4 seconds after flip start, controlled
    teardown). TWO MECHANISMS FROM THE SAME #1374 "OPTION 1" DAY, never
    reconciled: the slot COUNT changed, the slot SIZE did not.

    THE FIX IS "ALLOCATOR FOLLOWS THE PRICE", not the other direction: the
    layered 128 MiB/slot form is what ``BounceTerms.lane_buffer_bytes`` (and
    so ``total_bytes``, and so the host ledger's own charge) already prices,
    validated against boot weg2xsn31/3's own measured ARM line
    (``xchg_bounce=15.75`` GiB, exactly ``5 x (24 x 128 MiB) + staging``).
    Re-pricing the ledger on the REAL per-tag need instead (the other
    direction the operator named) would have to price P's most uneven PP
    stage as one unbroken tag -- correct, but it prices away almost all of
    the lever #1385 exists to be. This function is the one place that
    decision is made, so the ledger and the allocator can never drift apart
    on it again.

    Option 1 ACTIVE (``terms.max_tag_bytes > 0``): returns ``terms.
    slot_bytes`` -- the SAME oncard-slot unit ``BounceTerms.lane_buffer_bytes``
    already multiplies by ``terms.lane_slots`` to get the priced total, so
    ``leg_slot_bytes(terms) * terms.lane_slots == terms.lane_buffer_bytes``
    by construction, never a third number computed here.

    Option 1 ABSENT: :func:`leg_geometry`'s pre-#1374 pair, byte-identical --
    every boot that never stated ``max_tag_bytes`` is unaffected.
    """
    if int(getattr(terms, "max_tag_bytes", 0) or 0) > 0:
        return int(terms.slot_bytes)
    return int(leg_geometry(terms)[0])


# ---------------------------------------------------------------------------
# The carrier.
# ---------------------------------------------------------------------------


def bounce_path(boot_nonce: str, shm_root: str = xr.SHM_ROOT,
                lane: str = "") -> str:
    """The assemble buffer's file, one per boot and PER LANE.

    ONE FILE PER LANE, not one per card: the point of path (b) is that a unit
    is assembled ONCE and every destination reads its own rows out of that one
    copy, so a per-card file would be the image again, three times over. A
    LANE is one directed card pair (or the diagonal's card) -- the same unit a
    handshake is keyed by.

    THE LANE KEY IS THE DESK REPLAY'S OWN FINDING, and it is a product defect
    rather than a replay artefact. The file used to be keyed on the boot nonce
    ALONE while the handshake was keyed per pair, so the six ranks' concurrent
    legs serialised correctly against their OWN semaphores and then all wrote
    slot 0 of ONE shared buffer. Measured in the replay: every destination rank
    received well-formed bytes that belonged to another pair --
    ``first_diff=0``, `zero_bytes` ~0.4 % (i.e. real content, wrong content) --
    and the seam digest read MISMATCH on 6 of 8 tensors, deterministically.
    A boot would have shown the same as silently wrong weights, because
    nothing in the lane could notice: each pair's own protocol was obeyed.

    Empty ``lane`` keeps the historical single-buffer name, which is what the
    single-process ``phase=both`` form and every existing test use.
    """
    base = os.path.join(xr.region_dir(boot_nonce, shm_root),
                        xr.BOUNCE_SLOT_PREFIX)
    return base if not lane else f"{base}.{lane}"


def unlink_lane_buffer(boot_nonce: str, shm_root: str, lane: str) -> bool:
    """Free THIS lane's file -- #1385 step 3, boot weg2xsn31/6's own finding.

    ``LayerBounce.close()`` DELIBERATELY NEVER UNLINKS (this module's own
    docstring: cross-process store-and-forward needs the file to outlive
    either side's own ``close()``, since the depositor may finish and exit
    well before the collector even opens its mmap). That is correct and
    stays correct here -- but it also means a lane whose file is never
    explicitly removed keeps its tmpfs pages committed for the REST OF THE
    BOOT regardless of any concurrency cap, because nothing else unlinks a
    ``bounce.bin.<lane>`` file before full boot teardown
    (``weight_exchange_region.teardown_region``).

    THE GAP THIS CLOSES, MEASURED: on boot weg2xsn31/6, ``--xchg-lanes-
    concurrent 1`` correctly serialised PINNING -- never more than one
    ``LayerBounce`` being actively registered at a time (the #1385 step 2
    permit worked exactly as built) -- while the FILESYSTEM still showed
    FOUR lanes' files (``c0+p1+p2+p4``, 12.00 GiB against a promised
    3.00 GiB) coexisting, because the first three were each fully deposited
    AND collected, released their permit correctly, and then simply never
    went away. A concurrency cap that limits how many buffers may be PINNED
    at once is not the same claim as a cap on how many ACCUMULATE unfreed
    over a flip; by the time every lane has been touched once, the two
    converge to the SAME uncapped total regardless of the cap. This is the
    THIRD instance of "a number is priced, a different one is really held"
    on this one flag (#1358's per-lane undercount, #1385 step 1's slot-size/
    slot-count mismatch, and now this).

    CALLED ONLY AFTER THE COLLECTOR HAS POSTED ``drained`` FOR THIS TAG --
    the SAME moment that already tells the depositor it may reuse the
    buffer for the next tag (``CrossSlotRendezvous.wait_drained``). Unlinking
    any earlier would destroy bytes a collector on the other side of a real
    cross-process boot might not have read yet; unlinking here is safe
    because both this rank's own fd (closed inside ``run_bounce_leg``'s
    ``finally``, before this function is ever reached) and the depositor's
    fd (closed at the end of ITS OWN, earlier, ``run_bounce_leg`` call) are
    already closed by construction, so the kernel frees the pages the
    instant this call removes the last directory entry.

    Idempotent: an absent file is not an error -- the diagonal's other rank,
    or a retry, may already have removed it, and this function's caller
    (a single tag's single lane) has no way to tell those two cases apart
    from here, nor does it need to.
    """
    try:
        os.unlink(bounce_path(boot_nonce, shm_root, lane))
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# #1358 -- THE HOST-SLOT EMITTER, IN THE PRODUCTION PATH.
# ---------------------------------------------------------------------------
#
# IT EXISTED ONLY IN THE REPLAY SCRIPT, as a `print`, and never in product
# code -- so acceptance E2 could not have passed on ANY boot, and #1358's host
# ratchet (+1.2..1.7 GiB anon per P->D leg; xsn25 +4.65 GiB after ONE flip,
# then oom_kill) had no instrument at all. Built-but-not-wired, the class this
# campaign keeps paying for, this time in my own instrument.
#
# EVERY SLOT SAYS ITS BYTES AND ITS REGION AT ALLOCATION AND AT RELEASE, and
# the leg says what is still held when it ends: `bytes_live_after != 0` is the
# leak suspicion #1358 needs, stated as a number rather than inferred from a
# host sample.

HOST_SLOT_MARKER = "WEG2-XCHG-HOST-SLOT"
HOST_SLOT_LEG_MARKER = "WEG2-XCHG-HOST-SLOT-LEG"

#: Bytes this PROCESS currently holds in bounce slots, by path. A process-wide
#: registry and not a per-object counter: the question #1358 asks is "what does
#: this rank still hold", which no single buffer can answer.
_LIVE_SLOTS: Dict[str, int] = {}


def host_slot_live_bytes() -> int:
    """What this rank still holds in bounce slots, right now."""
    return sum(_LIVE_SLOTS.values())


def host_slot_line(*, event: str, path: str, nbytes: int, group: str = "",
                   rank: int = -1, leg: str = "", slot: int = -1,
                   region: str = "shm") -> str:
    """One line per slot per event. Every number carries its denominator."""
    return (
        f"{HOST_SLOT_MARKER} group={group or '?'} rank={rank} "
        f"leg={leg or '?'} slot={slot} event={event} bytes={int(nbytes)} "
        f"region={region} total_live_bytes={host_slot_live_bytes()} "
        f"path={path}"
    )


def _host_slot_event(event: str, path: str, nbytes: int, *, group: str = "",
                     rank: int = -1, leg: str = "", slot: int = -1,
                     region: str = "shm", log=None) -> None:
    """Record and announce one slot's allocation or release.

    THE REGISTRY IS UPDATED BEFORE THE LINE IS BUILT, so `total_live_bytes`
    is the state AFTER this event -- which is the number a reader wants when
    the event is a free.
    """
    if event == "alloc":
        _LIVE_SLOTS[path] = _LIVE_SLOTS.get(path, 0) + int(nbytes)
    else:
        if _LIVE_SLOTS.pop(path, None) is None:
            # A free with no matching alloc is itself a finding: the accounting
            # and the buffer have parted company.
            region = f"{region},unmatched-free"
    line = host_slot_line(event=event, path=path, nbytes=nbytes, group=group,
                          rank=rank, leg=leg, slot=slot, region=region)
    if log is not None:
        log(line)
    else:
        logger.info("%s", line)


def host_slot_leg_line(*, group: str = "", rank: int = -1, leg: str = "",
                       slots_before: int = 0) -> str:
    """The leg's own summary. `bytes_live_after != 0` is the leak suspicion."""
    return (
        f"{HOST_SLOT_LEG_MARKER} group={group or '?'} rank={rank} "
        f"leg={leg or '?'} slots_before={slots_before} "
        f"slots_live_after={len(_LIVE_SLOTS)} "
        f"bytes_live_after={host_slot_live_bytes()} -- a non-zero "
        f"bytes_live_after means this leg ended still holding host slots, "
        f"which is #1358's ratchet stated as a number instead of inferred "
        f"from a host sample"
    )


class LayerBounce:
    """``depth`` slots of ``slot_bytes`` of shm, registered, and DECLARED.

    Modelled on :class:`tp.HostBounce` deliberately rather than copied: same
    shm-file storage, same ``cudaHostRegister`` for the same measured 7.5 %,
    same declare-before-pin ordering, and the same contract that ``close()``
    DOES NOT UNLINK -- which is what makes a store-and-forward deposit
    possible across two process trees whose legs are different instants.

    THE BYTES ARE DECLARED TO THEIR OWNER BEFORE THEY ARE PINNED.
    ``pinned_host_budget`` is the #550 single owner of "may this pinned host
    buffer be allocated?", and this buffer is exactly the kind of post that
    registry exists for: non-swappable host memory on a box with no swap.  The
    #1269 launch-time ledger charges it too (``xchg_bounce_host_bytes``), and
    the two are not redundant -- that one bounds the reap watermark before the
    boot, this one answers at the moment of allocation.
    """

    POST_FLAG = BOUNCE_POST_FLAG

    @revert_pinned_posts_on_failure
    def __init__(self, ops: tp.DeviceOps, boot_nonce: str, *,
                 slot_bytes: int, depth: int, create: bool = True,
                 shm_root: str = xr.SHM_ROOT, lane: str = "",
                 group: str = "", rank: int = -1, leg: str = "",
                 slot_index: int = -1) -> None:
        if int(slot_bytes) <= 0:
            raise ValueError(f"slot_bytes must be positive, not {slot_bytes!r}")
        if int(depth) <= 0:
            raise ValueError(f"depth must be positive, not {depth!r}")
        self.ops = ops
        self.slot_bytes = int(slot_bytes)
        self.depth = int(depth)
        self.nbytes = self.slot_bytes * self.depth
        self.path = bounce_path(boot_nonce, shm_root, lane)
        # #1358 identity, carried so the emitter's line is readable without
        # joining it to anything: which rank, which leg, which slot.
        self._slot_group = str(group)
        self._slot_rank = int(rank)
        self._slot_leg = str(leg or lane)
        self._slot_index = int(slot_index)
        self._post = f"weg2-xchg-bounce {self.path}"
        check_and_register_pinned_post(self._post, self.POST_FLAG, self.nbytes)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self._fd = os.open(self.path, flags, 0o600)
        if create:
            os.ftruncate(self._fd, self.nbytes)
        self._mm = _mmap.mmap(self._fd, self.nbytes, _mmap.MAP_SHARED,
                              _mmap.PROT_READ | _mmap.PROT_WRITE)
        holder = ctypes.c_char.from_buffer(self._mm)
        self.ptr = ctypes.addressof(holder)
        del holder
        self._registered = False
        try:
            ops.host_register(self.ptr, self.nbytes,
                              tp.CUDA_HOST_REGISTER_PORTABLE)
            self._registered = True
        except Exception:  # noqa: BLE001 -- a 7.5 % regression, not a refusal
            self._registered = False
        # #1358: the slot is now held. `region` names WHERE the bytes are:
        # shm always, and `pinned` in addition once cudaHostRegister took.
        _host_slot_event("alloc", self.path, self.nbytes,
                         group=self._slot_group, rank=self._slot_rank,
                         leg=self._slot_leg, slot=self._slot_index,
                         region=("shm,pinned" if self._registered else "shm"))

    def slot_address(self, slot: int) -> int:
        return self.ptr + (int(slot) % self.depth) * self.slot_bytes

    def close(self) -> None:
        """Unpin, unmap, close the fd, and RELEASE THE POST.  Never unlink.

        The post goes last and unconditionally, for
        :meth:`tp.HostBounce.close`'s reason: a post that outlived its buffer
        charges the next admission for bytes nobody holds.
        """
        try:
            if self._registered:
                try:
                    self.ops.host_unregister(self.ptr)
                finally:
                    self._registered = False
            self._mm.close()
            os.close(self._fd)
        finally:
            unregister_pinned_post(self._post)
            _host_slot_event("free", self.path, self.nbytes,
                             group=self._slot_group, rank=self._slot_rank,
                             leg=self._slot_leg, slot=self._slot_index,
                             region="shm")


def registered_bounce_bytes() -> int:
    """Pinned host bytes currently DECLARED for bounce buffers, from #550's
    registry -- never from a flag this module sets about itself.

    The mutant this exists for ("the buffer is never released") cannot be
    caught by asking the leg whether it cleaned up; it is caught by asking the
    owner of the question.
    """
    return sum(int(p.nbytes) for p in registered_posts()
               if str(p.name).startswith("weg2-xchg-bounce "))


# ---------------------------------------------------------------------------
# Path (a): which descriptors qualify, and nothing more.
# ---------------------------------------------------------------------------


@dataclass
class BounceResult:
    """One leg's own accounting, with every number naming its instrument.

    ``verdict`` IS AN ACCOUNTING VERDICT, NOT A BYTE COMPARISON, and the
    distinction is deliberate: ``MATCH`` says every band this leg deposited was
    collected in full and the plan's byte total was moved.  It does NOT say the
    destination bytes equal the source bytes -- nothing on this path can say
    that without reading the destination back, which would double the traffic
    the whole slice exists to bound.  The byte proof lives in two other places
    and is named here so this field is never mistaken for it: path (a)'s slot
    checksums on the metal, and the hermetic smoke's own comparison against the
    source rows (``test_weg2_xchg_bounce_execution_smoke_1273``).
    """

    units: int
    bands: int
    deposited_bytes: int
    collected_bytes: int
    planned_bytes: int
    host_bytes_peak: int
    slot_bytes: int
    depth: int
    widest_unit_key: Tuple[str, str]
    widest_unit_bytes: int
    widest_run_bytes: int
    overlap: str
    deposit_ms: float = 0.0
    collect_ms: float = 0.0
    short: Tuple[str, ...] = field(default_factory=tuple)
    #: The SHADOW-COMPARE verdict, or None on the authoritative path where
    #: nothing is compared.  Carried on the result rather than logged and
    #: dropped, because the per-boot summary is over the legs' verdicts.
    inject: Optional["InjectVerdict"] = None
    #: THE MODE THIS LEG RAN, carried from the one place that validated it
    #: (`run_bounce_leg`, which REFUSES an unknown mode rather than defaulting
    #: -- "a default here would decide whether this leg owns 27 GiB of
    #: weights").  #1336: the summary line used to INFER this from
    #: `inject is None`, which is a SECOND decision on the question the module
    #: states it refuses to decide twice, and it inferred in the DANGEROUS
    #: direction: `authoritative`.  The inference happened to be faithful at
    #: the single construction site below -- `comparing = mode ==
    #: INJECT_SHADOW`, so `inject is None` did mean authoritative there -- but
    #: it is the FIELD DEFAULT of a plain dataclass, so any other
    #: construction (an error path, a future early return, a test double)
    #: printed `mode=authoritative` for a leg that never ran authoritative.
    #: An empty string prints as `unset`: a named state, never a lie in the
    #: direction that costs 27 GiB.
    mode: str = ""
    #: Units that did NOT fit one depth-slot and were therefore banded.  By
    #: the AMENDMENT 3 decide these may only ever be UNLAYERED classes
    #: (``lm_head`` takes 4 bands at a 721 MiB slot); a LAYER among them is a
    #: refusal, not a log line, so this list existing is not an escape hatch --
    #: it is how "which classes are not resident whole" stays answerable from
    #: the log instead of being argued.
    banded: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        if self.short:
            return "SHORT"
        if not (self.deposited_bytes == self.collected_bytes
                == self.planned_bytes):
            return "ACCOUNTING-DISAGREE"
        return "MATCH"

    def line(self) -> str:
        """The RUN-moment line.  Deliberately NOT ``WEG2-XCHG-BOUNCE``.

        ``xchg_bounce.arm_line`` already owns that prefix and prints the ARM
        moment: the sizing expression, the coverage verdict and Sigma H.  This
        line reports what the leg then MOVED.  Two moments, two prefixes: a
        single prefix carrying both would make an arm figure and a run figure
        indistinguishable to any parser, which is the instrument confusion
        #1005 was built to end.

        Section 10.2's rule still binds -- the terms are printed, not just the
        total -- because a boot whose cut differs gets a different number from
        the same formula and a bare total could not be checked against it.
        """
        return (
            "WEG2-XCHG-BOUNCE-LEG "
            f"units={self.units} bands={self.bands} "
            f"widest_unit={self.widest_unit_key[1]} "
            f"widest_unit_bytes={self.widest_unit_bytes} "
            f"widest_run_bytes={self.widest_run_bytes} "
            f"slot_bytes={self.slot_bytes} depth={self.depth} "
            f"assemble_bytes={self.slot_bytes * self.depth} "
            f"bounce_total_bytes={self.host_bytes_peak} "
            f"deposited={self.deposited_bytes} collected={self.collected_bytes} "
            f"planned={self.planned_bytes} "
            f"deposit_ms={self.deposit_ms:.1f} collect_ms={self.collect_ms:.1f} "
            f"mode={self.mode or wx.INJECT_MODE_UNSET} "
            # ALWAYS PRINTED, NEVER OMITTED. The field used to vanish when
            # `inject` was None, and an absent field is a silence a reader
            # fills in: weg2xsn25 read three legs' MISSING phase as proof they
            # ran unsplit when they had run collect. Same shape here, so the
            # same rule -- say which of the two silences this is:
            #   by-design-authoritative: no comparison EXISTS under this mode
            #                            (`comparing = mode == INJECT_SHADOW`)
            #   NOTHING-COMPARED:        a SHADOW leg that graded nothing,
            #                            which is a finding and not a state
            # #1336 THREE SILENCES, NOT TWO. An ABSENT mode used to fall into
            # the `NOTHING-COMPARED` arm, i.e. a line with no recorded mode
            # claimed a SHADOW leg had graded nothing -- a finding invented out
            # of missing evidence. `mode=` one field to the left already says
            # `unset` in that case; this field now agrees with it instead of
            # contradicting it.
            + (f"inject={self.inject.verdict} " if self.inject else
               ("inject=by-design-authoritative "
                if str(self.mode) == wx.INJECT_AUTHORITATIVE
                else "inject=NOTHING-COMPARED "
                if str(self.mode) == wx.INJECT_SHADOW
                else f"inject={wx.INJECT_MODE_UNSET} "))
            + f"banded={len(self.banded)}"
            + (f" banded_units={','.join(self.banded)}" if self.banded else "")
            + f" overlap={self.overlap} verdict={self.verdict}"
        )


@dataclass
class InjectVerdict:
    """One leg's SHADOW-COMPARE result: assembled bytes vs refilled weights.

    THIS IS A BYTE VERDICT, unlike :attr:`BounceResult.verdict`, and the
    difference is why S6I boots `shadow` first.  The accounting verdict says
    every band was moved; this says the bytes the exchange would have served
    ARE the bytes the disk refill actually produced.  A transfer never graded
    against a known-correct copy of the same bytes is not evidence, however
    clean its accounting.

    ``mismatch_first`` names the DESCRIPTOR, not the byte: the next question
    after MISMATCH is always "which tensor", and a byte offset without the
    parameter name is a postmortem nobody can start from.
    """

    mode: str
    pieces: int
    bytes_compared: int
    mismatches: int
    mismatch_first: str = ""
    rows_compared: int = 0
    #: WHICH HALF OF THE HANDSHAKE THIS LEG RAN. The line carried no phase at
    #: all, and boot weg2xsn25 shows what that costs: three P legs were read as
    #: "ran without a phase, therefore unsplit" when they had in fact run
    #: `collect` -- the W68 texts they carried are reachable ONLY under
    #: `rendezvous is not None and phase == PHASE_COLLECT` (:1176). The ABSENCE
    #: of a field is not evidence about the value; it is evidence about the
    #: instrument. Empty means the unsplit form, which is now visible as such.
    phase: str = ""

    @property
    def verdict(self) -> str:
        if self.pieces <= 0:
            # NOTHING WAS COMPARED, and that may not read as MATCH.  A leg
            # whose plan was empty, or whose compare was skipped, produced NO
            # EVIDENCE -- and no evidence printed as MATCH is exactly the
            # instrument that cannot fail.
            return "NO-COMPARE"
        return "MATCH" if self.mismatches == 0 else "MISMATCH"

    def line(self) -> str:
        return (
            "WEG2-XCHG-INJECT "
            f"mode={self.mode} phase={self.phase or 'unsplit'} "
            f"verdict={self.verdict} "
            f"pieces={self.pieces} bytes={self.bytes_compared} "
            f"rows={self.rows_compared} mismatches={self.mismatches} "
            f"mismatch_first={self.mismatch_first or '-'}"
        )


#: EVERY COMPARED LEG OF THIS PROCESS, in order, for the per-BOOT summary.
#:
#: #1342 S3.  `inject_summary_line` answers "has this boot ever disagreed",
#: which needs every leg -- and it had NO production caller at all, so the
#: question was never once asked on any boot.  A per-leg emitter cannot answer
#: it, so the legs accumulate here and the summary is re-emitted after each
#: compared leg: a running answer, which is the right shape for a question
#: about the boot so far.  Only COMPARED legs are appended (`mode=shadow`);
#: an authoritative leg produces no verdict and must not be counted as a
#: `NO-COMPARE`, which would make the arm that owns the bytes look like the
#: arm that stopped grading.
_INJECT_VERDICTS: List["InjectVerdict"] = []


def reset_inject_verdicts() -> None:
    """Drop the accumulated verdicts.  For tests, and for a boot that re-arms.

    Exposed rather than left to attribute surgery on the module: a test that
    reached into `_INJECT_VERDICTS` directly would be the second writer of the
    same state.
    """
    _INJECT_VERDICTS.clear()


def inject_summary_line(verdicts: Sequence["InjectVerdict"]) -> str:
    """The per-BOOT summary over every leg's verdict.

    Separate from the per-leg line because the two answer different questions
    -- "did THIS leg agree" and "has this boot ever disagreed" -- and a boot
    that matched 35 legs and mismatched one must not read as a pass.
    ``legs_no_compare`` is printed rather than folded into either side: a leg
    that compared nothing is neither a match nor a mismatch, and hiding it
    would let an arm that silently stopped comparing look perfect.
    """
    total = len(verdicts)
    mism = [v for v in verdicts if v.verdict == "MISMATCH"]
    none = [v for v in verdicts if v.verdict == "NO-COMPARE"]
    first = next((v.mismatch_first for v in mism if v.mismatch_first), "-")
    return (
        "WEG2-XCHG-INJECT-SUMMARY "
        f"legs={total} legs_match={total - len(mism) - len(none)} "
        f"legs_mismatch={len(mism)} legs_no_compare={len(none)} "
        f"pieces={sum(v.pieces for v in verdicts)} "
        f"bytes={sum(v.bytes_compared for v in verdicts)} "
        f"mismatch_first={first} "
        f"verdict={'MATCH' if (total and not mism and not none) else 'NOT-CLEAN'}"
    )


def _deposit_band(ops: tp.DeviceOps, stream: int, descs: Sequence[object],
                  batch, base: int) -> int:
    """COMPACTING D2H: the slot holds payload only, so the link carries no
    padding.  ``dpitch`` is the run -- ``tp.run_producer_pair``'s own
    convention, reused rather than re-derived so the two lanes cannot drift.
    """
    moved = 0
    for piece in batch.pieces:
        desc = descs[piece.desc_index]
        src_ptr = int(desc.src_ptr) + piece.src_off
        if piece.kind == wx.FLAT:
            ops.memcpy_async(base + piece.slot_off, src_ptr, piece.nbytes,
                             stream)
        else:
            ops.memcpy2d_async(base + piece.slot_off, piece.run_bytes,
                               src_ptr, piece.spitch,
                               piece.run_bytes, piece.rows, stream)
        moved += int(piece.nbytes)
    return moved


def _collect_band(ops: tp.DeviceOps, stream: int, descs: Sequence[object],
                  batch, base: int) -> int:
    """SCATTERING H2D: ``spitch`` is the run (the slot is compact), ``dpitch``
    is the destination arena's pitch -- ``tp.run_consumer_pair``'s convention.

    THIS is "jede karte nimmt sich von dem was er braucht": the piece's
    ``dst_off``/``dpitch`` are the destination's OWN rows, and a piece whose
    band is not this destination's is simply not in its descriptor list.
    """
    moved = 0
    for piece in batch.pieces:
        desc = descs[piece.desc_index]
        dst_ptr = int(desc.dst_ptr) + piece.dst_off
        if piece.kind == wx.FLAT:
            ops.memcpy_async(dst_ptr, base + piece.slot_off, piece.nbytes,
                             stream)
        else:
            ops.memcpy2d_async(dst_ptr, piece.dpitch,
                               base + piece.slot_off, piece.run_bytes,
                               piece.run_bytes, piece.rows, stream)
        moved += int(piece.nbytes)
    return moved


def _compare_band(ops: tp.DeviceOps, stream: int, descs: Sequence[object],
                  batch, base: int, scratch: int,
                  verdict: "InjectVerdict") -> None:
    """SHADOW COMPARE: the LIVE destination rows against the staged bytes.

    The staged bytes are already in the slot at ``base`` -- compacted, payload
    only -- so the compare pulls the destination's CURRENT content (what the
    refill just wrote) into ``scratch`` with the SAME compaction and memcmps.
    One extra host slot and one D2H per band; no device scratch at all, which
    is the point of doing it through the buffer that is already mapped.

    IT COMPARES WHAT THE AUTHORITATIVE PATH WOULD HAVE WRITTEN: the same
    ``piece`` geometry :func:`_collect_band` writes with is what is read back,
    so a row-map defect shows here instead of being cancelled out by reading
    with the same wrong map.
    """
    for piece in batch.pieces:
        desc = descs[piece.desc_index]
        dst_ptr = int(desc.dst_ptr) + piece.dst_off
        if piece.kind == wx.FLAT:
            ops.memcpy_async(scratch + piece.slot_off, dst_ptr, piece.nbytes,
                             stream)
        else:
            ops.memcpy2d_async(scratch + piece.slot_off, piece.run_bytes,
                               dst_ptr, piece.dpitch,
                               piece.run_bytes, piece.rows, stream)
    ops.synchronize(stream)
    for piece in batch.pieces:
        desc = descs[piece.desc_index]
        n = int(piece.nbytes)
        staged = ctypes.string_at(base + piece.slot_off, n)
        live = ctypes.string_at(scratch + piece.slot_off, n)
        verdict.pieces += 1
        verdict.bytes_compared += n
        verdict.rows_compared += int(piece.rows)
        if staged != live:
            verdict.mismatches += 1
            if not verdict.mismatch_first:
                verdict.mismatch_first = (
                    f"{getattr(desc, 'param_name', '?')}"
                    f"@dst_rank={getattr(desc, 'dst_rank', '?')}"
                    f"+{piece.dst_off}")


#: #1330 B4n. THE THREE PHASES OF A BOUNCE LEG.
#:
#: THE ROOT OF weg2xsn20's 24/24 ``verdict=NO-COMPARE``, and it was never a
#: missing pointer: this function's caller ran DEPOSIT and (COMPARE|COLLECT)
#: over ONE descriptor list in ONE process in ONE pass (the loop below,
#: :func:`run_bounce_leg`), so it needed ``src_ptr`` AND ``dst_ptr`` as DEVICE
#: addresses simultaneously.  At the destination hook ``src_ptr`` is by
#: construction the PEER's address (``weight_exchange_shadow.py:3336-3344``
#: picks the side from the hook at ``:3329-3331``), which no process can read.
#: Hence ``src_resolved=0/N dst_resolved=N/N`` on every leg, never the mirror.
#:
#: Split into phases, each side needs only the pointer IT owns:
#: the depositing rank reads from its own weights, the collecting rank writes
#: into its own pages, and the host slot is the medium between them.
PHASE_DEPOSIT = "deposit"
PHASE_COLLECT = "collect"
PHASE_BOTH = "both"
PHASE_CHOICES = (PHASE_DEPOSIT, PHASE_COLLECT, PHASE_BOTH)


def _missing_pointer(descs: Sequence[object],
                     phase: str = PHASE_BOTH) -> Optional[str]:
    """W74's case, PER PHASE: the pointer the phase actually needs.

    Assembly does not CREATE a source, it only stages one, so a descriptor with
    no pointer on the side the phase reads is a hole in the plan and must be
    named, never zeroed.  Which side that is depends on the phase, and treating
    ``both`` as the universal answer is precisely what turned an unreachable
    PEER address into 24 refusals:

    * ``deposit`` reads ``src_ptr`` (this rank's own weights) and writes the
      slot -- it needs NO destination address at all;
    * ``collect`` reads the slot and writes ``dst_ptr`` (this rank's own pages)
      -- it needs NO source address;
    * ``both`` is the single-process form and needs both, unchanged.
    """
    need_src = str(phase) in (PHASE_DEPOSIT, PHASE_BOTH)
    need_dst = str(phase) in (PHASE_COLLECT, PHASE_BOTH)
    for d in descs:
        if need_src and getattr(d, "src_ptr", None) is None:
            return f"{getattr(d, 'param_name', '?')} has no source pointer"
        if need_dst and getattr(d, "dst_ptr", None) is None:
            return f"{getattr(d, 'param_name', '?')} has no destination pointer"
    return None


#: THE PHASE-ORDERING REFUSAL IS W68's, AND IT GETS NO NAME OF ITS OWN.
#:
#: A first draft declared ``Weg2XchgBouncePhaseUnordered(RuntimeError)`` and
#: wrote ``W68 Weg2XchgPlanDisagree:`` into its messages.
#: ``test_weg2_wcode_uniqueness_1263`` caught it immediately --
#: ``{'W68': {'Weg2XchgBouncePhaseUnordered', 'Weg2XchgPlanDisagree'}}`` -- and
#: it was right: the census reads the MESSAGE TEXT (``ASSIGNMENT`` at :96), so
#: a second name beside a code is a collision whether or not a class exists.
#: The plan record's rule stands (no new W-code; W23/W39 stay free), and this
#: condition IS W68's own sentence: the two ends disagree about what is in the
#: slot.  The alias keeps the call site readable without minting a second
#: holder.
#:
#: THE DANGER DIRECTION IT NAMES, kept here because the alias has no docstring:
#: reading a slot before its producer posted ``full`` returns whatever the
#: previous band left there -- the destination is then served plausible bytes
#: from the WRONG layer, with every counter green.  That is silent corruption
#: and strictly worse than any refusal, so the collect verifies the handshake
#: BEFORE its first copy.  Same reasoning as ``run_consumer_pair``'s
#: short-piece check (``weight_exchange_transport.py:1754``: *"raises W70 and
#: issues NOTHING"*), applied to the bounce's own slots.
Weg2XchgBouncePhaseUnordered = wx.Weg2XchgPlanDisagree


def _require_rendezvous(phase: str, rendezvous) -> None:
    """A phased leg without a handshake is refused, never run optimistically.

    ``both`` is the single-process form and needs none.  ``deposit`` and
    ``collect`` run in DIFFERENT processes over a shared host slot, so a leg
    that skipped the handshake would be exactly the unordered read above.
    """
    if str(phase) == PHASE_BOTH:
        return
    if rendezvous is None:
        raise Weg2XchgBouncePhaseUnordered(
            f"W68 Weg2XchgPlanDisagree: phase={phase} needs a slot "
            f"handshake and none was given. The depositing and collecting "
            f"ranks are different processes sharing a host slot; running "
            f"either without the empty/full protocol would let a collect read "
            f"a band no deposit has written, which is silent corruption and "
            f"not a refusal."
        )


def run_bounce_leg(
    descs: Sequence[object],
    ops: tp.DeviceOps,
    boot_nonce: str,
    *,
    slot_bytes: Optional[int] = None,
    depth: Optional[int] = None,
    terms: Optional[xb.BounceTerms] = None,
    #: DEFAULTS TO `shadow`, the SAFE mode, matching the product's own default
    #: (`--weg2-xchg-inject`).  A function whose default writes the live
    #: weights while the flag's default does not is a trap: every caller that
    #: forgets the argument takes the dangerous path, and the one that matters
    #: is the product.  Authoritative must be asked for.
    mode: str = wx.INJECT_SHADOW,
    shm_root: str = xr.SHM_ROOT,
    device: int = 0,
    #: #1378 xsn36 (the lock-ordering deadlock): THIS rank's NVML uuid. When
    #: set, run_bounce_leg takes the per-card PCIe serialisation lock around
    #: EACH COPY (the D2H band write, the H2D band read) and holds it NEVER
    #: across a rendezvous wait -- the co-located pair on one physical card
    #: (weg2xsn35/36: P rank0 + D rank0 on the 5090) deadlocked by
    #: construction when the CALLER held the card lock across the whole leg
    #: while the leg's waits needed the SIBLING's posts, which the sibling
    #: could not produce without the same lock. Waits outside, copies inside.
    pcie_uuid: Optional[str] = None,
    pcie_direction: Optional[str] = None,
    #: #1378 xsn36/37 (coordinator requirement (a)): called between the
    #: chunked waits of the COLLECT phase; False means the co-located deposit
    #: rank is GONE -- the leg refuses NOW, named, instead of after a silent
    #: budget burn.  Fail-open by contract: the callback answers True on any
    #: instrument error, and the 120 s budget remains the detector.
    liveness=None,
    #: #1330 B4n.  WHICH HALF OF THE BOUNCE THIS RANK RUNS.  ``both`` is the
    #: single-process form and is byte-identical to the behaviour before the
    #: split; ``deposit``/``collect`` are the two ranks of a cross-group leg.
    phase: str = PHASE_BOTH,
    #: #1358 identity for the host-slot lines. Defaulted so every existing
    #: caller and test is unchanged; the product passes them from the adapter,
    #: which is the only place group and rank both exist.
    leg_group: str = "",
    leg_rank: int = -1,
    leg_name: str = "",
    #: WHICH LANE's buffer this leg assembles in -- one directed card pair, or
    #: the diagonal's card. Empty keeps the single shared buffer, which is the
    #: single-process `both` form. See :func:`bounce_path` for the measured
    #: reason a phased leg may never share one.
    lane: str = "",
    #: The slot handshake, INJECTED rather than constructed here.  The product
    #: passes an adapter over the region's CROSS semaphores (which already
    #: exist -- weg2xsn20's teardown census counted 24 of them beside the 12
    #: diagonal); a test passes a double it can drive OUT OF ORDER, which is
    #: the only way to prove the refusal can fire.  ``None`` is legal only for
    #: ``both``.
    rendezvous=None,
    log=None,
) -> BounceResult:
    """Assemble every unit in the bounded buffer; every card takes its rows.

    THE PIPELINE IS WHAT DEPTH IS FOR, and it is implemented rather than
    asserted.  With ``depth >= 2`` the collect of band *n* is left IN FLIGHT on
    its own stream while band *n+1* is deposited into the other slot, and a
    slot is only re-deposited after its own collect has been synchronised --
    which is the section 10.4 requirement ("assemble(layer k+1) overlaps
    copy-out(layer k)") expressed as the order of two streams rather than as a
    comment.  At ``depth == 1`` there is no other slot, so the collect must be
    drained before the next deposit and ``overlap`` reports ``none``: a
    correct transfer and a degraded one, and the line says which.

    ``slot_bytes`` IS NOT DERIVED HERE.  Pass ``terms`` (an
    ``xchg_bounce.BounceTerms``) and the geometry comes from
    :func:`leg_geometry`, i.e. from the ARM's own priced decision; the explicit
    ``slot_bytes``/``depth`` pair exists for tests and tools that size by hand.
    The sizing expression and its ledger charge have ONE owner
    (``xchg_bounce`` + the launcher's ``xchg_bounce_host_bytes``), and a second
    derivation inside the transport would be the two-ledgers defect this
    campaign has already paid for.  What this function owns is the two
    REFUSALS that the number cannot work for THIS plan
    (:func:`refuse_if_plan_exceeds_slot`, :func:`refuse_if_slot_short`).

    THE ROW MAP IS NOT COMPUTED HERE EITHER, and that is the S1 boundary the
    spec's two prior incidents demand.  Every offset this loop uses is the
    descriptor's own ``src_off``/``dst_off``/``spitch``/``dpitch`` as
    ``weight_exchange.build_plan`` emitted them -- which is where
    ``device_block_offsets`` already knows that ``qkv_proj`` is three device
    sub-blocks and ``in_proj_qkvz`` FOUR at rows ``0 / g_k / 2*g_k /
    2*g_k+g_v`` (``qwen3_5.py:543``) and not the checkpoint's byte offsets.
    A second offset derivation here is precisely the defect that was paid for
    twice, so there is none.
    """
    # #1385/xsn31-4: THE ONE BOOLEAN, computed here and reused at every site
    # below that must agree with it (the slot size, the AMENDMENT-2 refusal,
    # the slot COUNT) -- three sites spelling this predicate three different
    # ways is exactly how the slot-size/slot-count mismatch this fixes
    # reached the metal unnoticed.
    _option1_leg = (terms is not None
                    and int(getattr(terms, "max_tag_bytes", 0) or 0) > 0)
    # #1397 OPTION 3: THE ONE BOOLEAN for band-credit reuse, same doctrine as
    # `_option1_leg` above -- one predicate, reused at every site that must
    # agree with it (the slot COUNT below, the wrap check, the collector's
    # post). CROSS PAIRS ONLY (`rendezvous.pair is not None`): see
    # DESIGN_option3_band_credit_0914.md -- a diagonal lane's own collector's
    # `resume` is gated on THIS rank's not-yet-published VRAM credit, so a
    # wait for that collector's drain, inside this rank's own deposit loop,
    # before that credit is published, is the exact cycle #1374 deleted.
    # `getattr` on both `terms` and `rendezvous` because neither is
    # guaranteed non-None here (this predicate is computed before either is
    # validated further down), and an absent attribute must read as "not
    # eligible", never raise.
    _band_credit_leg = (_option1_leg
                        and bool(getattr(terms, "band_credit", False))
                        and getattr(rendezvous, "pair", None) is not None)
    if terms is not None:
        derived_slot = leg_slot_bytes(terms)
        derived_depth = leg_geometry(terms)[1]
        slot_bytes = derived_slot if slot_bytes is None else slot_bytes
        depth = derived_depth if depth is None else depth
    if slot_bytes is None or depth is None:
        # NAMED, BECAUSE IT COST A WHOLE BOOT UNSEEN. This was a bare
        # `ValueError`, and on weg2xsn26 it fired nine times on D's sleep leg
        # and ended the boot at W17 Weg2GroupDead -- visible ONLY because W29
        # Weg2FlipRankDisagree collected the foreign text. A refusal on a path
        # that costs a boot may not be the one refusal without a code: the
        # census scans for W-codes, and an unnamed raise is invisible to it by
        # construction.
        #
        # W68 AND NOT A NEW CODE: this is the plan and the arm disagreeing
        # about a size, which is exactly `Weg2XchgPlanDisagree`'s subject, and
        # the census reads the MESSAGE TEXT rather than the class.
        raise wx.Weg2XchgPlanDisagree(
            "W68 Weg2XchgPlanDisagree: run_bounce_leg needs either `terms` "
            "(the ARM's priced decision) or an explicit slot_bytes/depth "
            "pair; it derives no size of its own, because the sizing "
            "expression has one owner (xchg_bounce). The caller that reaches "
            "here has a plan but no price, so it would have to invent one."
        )
    descs = list(descs)
    phase = str(phase)
    if phase not in PHASE_CHOICES:
        raise ValueError(
            f"unknown bounce phase {phase!r}; one of {PHASE_CHOICES}. Refused "
            f"rather than defaulted: a default here would decide whether this "
            f"rank writes into the live weights or merely stages bytes")
    _require_rendezvous(phase, rendezvous)
    hole = _missing_pointer(descs, phase)
    if hole is not None:
        # #1345 (a''): THE CENSUS RIDES THE REFUSAL, not just the first
        # offender.  `hole` names one parameter and one side -- that is (a') and
        # it stays -- but only the counts distinguish "one descriptor is a hole
        # in an otherwise resolvable plan" from "this lane has no source at all
        # on any descriptor", and boots weg2xsn18/19 each paid a window for that
        # ambiguity.  Computed from descriptors: no byte moves here either.
        _prof = wx.pointer_profile(descs)
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing bounce: {hole} -- assembly stages a "
            f"source, it does not create one, so this slice would be served "
            f"with undefined bytes. Refusing before the buffer is mapped. "
            f"PROFILE mode={mode} phase={phase} "
            f"src_resolved={_prof.src_resolved}/{_prof.descs_total} "
            f"dst_resolved={_prof.dst_resolved}/{_prof.descs_total} "
            f"pieces={_prof.pieces_total} -- an unresolved side means the leg "
            f"has no ADDRESS there, never that the bytes differ"
        )
    refuse_if_slot_short(slot_bytes, descs)
    if not _option1_leg:
        # AMENDMENT 2: a unit is assembled COMPLETE in one depth-slot.  This
        # is the run-moment half of the ARM's coverage grade -- see the
        # function's docstring for why one check cannot cover both moments.
        #
        # SKIPPED UNDER OPTION 1 (#1385/xsn31-4), and that is not a hole:
        # Amendment 2's invariant is "one LAYER, one slot", sized on
        # `slot_bytes = widest_layer_bytes` so it always held trivially. Under
        # Option 1 `slot_bytes` is the much smaller 128 MiB oncard unit and a
        # layer is MEANT to band across many of them within the same
        # tag-sized buffer -- the guarantee moved from "one layer, one slot"
        # to "one TAG, one buffer" (Option 1's own design, BounceSlots'
        # docstring). Calling this check here too would refuse every
        # Option-1 boot on its very first layer. The equivalent protection
        # for Option 1 is the in-loop `bands >= slots` refusal a few lines
        # below (W68 Weg2XchgPlanDisagree), graded against the SAME `slots`
        # this buffer is actually built with.
        refuse_if_plan_exceeds_slot(slot_bytes, descs, terms)

    units = plan_units(descs)
    if not units:
        raise wx.Weg2XchgSourceMissing(
            "W74 Weg2XchgSourceMissing bounce: the plan contains no unit with "
            "a source -- there is nothing to assemble, and a leg that "
            "reported success here would report a flip that moved no weights."
        )
    widest = widest_unit(units)
    planned = sum(u.nbytes for u in units)
    # By the AMENDMENT 3 decide these are UNLAYERED classes only -- a LAYER
    # above the slot was already refused above, so this list can never carry
    # one, and the test that pins that is
    # `test_a_layer_above_the_slot_refuses_but_an_unlayered_class_bands`.
    banded = tuple(u.key[1] for u in units
                   if u.nbytes > int(slot_bytes))

    # THE MODE IS READ ONCE, HERE, and carried -- never re-read at the branch.
    # A second read could see a different answer than the one this leg was
    # entered with, and the two answers differ by "does this write into the
    # live weights".
    mode = str(mode)
    if mode not in wx.INJECT_CHOICES:
        raise ValueError(
            f"unknown inject mode {mode!r}; one of {wx.INJECT_CHOICES}. "
            f"Refused rather than defaulted: a default here would decide "
            f"whether this leg owns 27 GiB of weights")
    comparing = mode == wx.INJECT_SHADOW
    verdict = InjectVerdict(mode=mode, pieces=0, bytes_compared=0,
                            mismatches=0, phase=str(phase))
    # SHADOW MODE COSTS ONE EXTRA SLOT, priced rather than borrowed: the
    # compare needs the live bytes beside the staged ones, and reusing a
    # depth-slot would overwrite the band still in flight.
    # ONE AUTHORITY (xchg_bounce.assemble_slots): this line and the launcher's
    # price were two expressions for one buffer, and the difference was exactly
    # one widest layer under `mode=shadow` -- 721.4 MiB the ledger never
    # charged, on every S6I boot.
    # #1374 OPTION 1: the slot count comes from the TERMS, one producer
    # (`xb.tag_slots` via `BounceTerms.lane_slots`), so the file this leg
    # allocates is the file the ledger charged for. Without terms -- or with a
    # boot that did not state `max_tag_bytes` -- this is the old depth-sized
    # floor and the wrap refusal below is what keeps it honest.
    # #1397 OPTION 3: a CROSS lane with band credit active allocates
    # `terms.cross_lane_slots` instead -- the SAME producer discipline,
    # `BounceTerms` is still the one authority, just a different property of
    # it for a lane the whole-tag floor was never required for.
    #
    # AN if/elif/else, NOT A TERNARY -- pyright cannot narrow `terms` past
    # `Optional[BounceTerms]` through a bare `_band_credit_leg`/`_option1_leg`
    # boolean (it is not an inline `terms is not None` expression), so a
    # ternary here reads as a possible `None.cross_lane_slots` even though
    # both booleans PROVE `terms is not None` by their own definitions
    # above (`_option1_leg = terms is not None and ...`,
    # `_band_credit_leg = _option1_leg and ...`). The `assert` in each
    # branch is that proof, spelled for the type checker in the same
    # branch it is read in -- pinned rather than silenced.
    if _band_credit_leg:
        assert terms is not None
        slots = int(terms.cross_lane_slots)
    elif _option1_leg:
        assert terms is not None
        slots = int(terms.lane_slots)
    else:
        slots = xb.assemble_slots(int(depth), comparing=comparing)
    if phase != PHASE_BOTH and not lane:
        raise Weg2XchgBouncePhaseUnordered(
            f"W68 Weg2XchgPlanDisagree: phase={phase} without a lane key. The "
            f"assemble buffer would be shared with every other pair running "
            f"concurrently, and each pair's own handshake would still be "
            f"obeyed -- so the corruption is silent. Refusing.")
    # #1358: the slot census is taken across THIS leg, so the before-count is
    # read here and the after-count in the `finally` below.
    _slots_before = len(_LIVE_SLOTS)
    bounce = LayerBounce(ops, boot_nonce, slot_bytes=slot_bytes, depth=slots,
                         create=True, shm_root=shm_root, lane=lane,
                         group=str(leg_group), rank=int(leg_rank),
                         leg=str(leg_name or lane), slot_index=0)
    d_stream = ops.create_stream(device)
    c_stream = ops.create_stream(device)
    #: What each slot is still draining, so a slot is never re-deposited under
    #: a collect that has not landed.  This list IS the pipeline's state.
    # #1374: THE FILE'S SLOT COUNT, not `depth`. `slots` is what was
    # allocated (assemble_slots adds the shadow's compare slot) while the
    # loop indexed by `depth`, so at the #1358 default depth=1 every band
    # landed in slot 0: no double buffering at all, and the ledger charged
    # for a slot the loop could not reach.
    inflight: List[Optional[object]] = [None] * int(slots)
    if _band_credit_leg and phase == PHASE_DEPOSIT:
        # `_band_credit_leg` proves `rendezvous is not None` (its own
        # definition above: `getattr(rendezvous, "pair", None) is not
        # None`) -- spelled for the type checker, same doctrine as the
        # `slots` if/elif/else two screens up.
        assert rendezvous is not None
        # #1397: discard a PRIOR tag's uncollected tail before this tag's own
        # first band -- see `CrossSlotRendezvous.prime_band_drain`'s own
        # docstring for why this is the only safe instant.
        rendezvous.prime_band_drain()
    deposited = collected = 0
    bands = 0
    deposit_ms = collect_ms = 0.0
    short: List[str] = []
    try:
        for unit in units:
            for batch in tp.batch_descs(list(unit.descs), int(slot_bytes)):
                slot = bands % int(slots)
                # #1374 OPTION 1's PRECONDITION, ENFORCED HERE. Removing the
                # per-band claim is only safe while every band of this tag owns
                # its own slot; the moment the count wraps, band N would
                # overwrite band 0 while the collector may still be reading it
                # -- silently, which is worse than the deadlock it replaced.
                # The per-tag `drained` handshake covers the NEXT tag; this
                # covers the bands WITHIN one. Until the terms carry
                # max_tag_bytes and size the buffer accordingly, this refuses
                # by name rather than pricing a wrap nobody can see.
                if int(bands) >= int(slots) and phase == PHASE_DEPOSIT:
                    if _band_credit_leg:
                        # #1397 OPTION 3: THE QUITTUNG AS PRECONDITION, not a
                        # removal of the guard above -- a CROSS lane may wait
                        # here because its collector's readiness never
                        # depends on THIS rank (see the design doc); a
                        # diagonal lane can never reach this branch
                        # (`_band_credit_leg` requires `rendezvous.pair is
                        # not None`), so it still takes the `else` refusal
                        # below, unchanged, exactly as before #1397.
                        #
                        # PROVEN, NOT HOPED: `_band_credit_leg` is False
                        # whenever `rendezvous` is None (its own definition
                        # above reads `getattr(rendezvous, "pair", None)`,
                        # which answers `None` rather than raising on a
                        # None receiver) -- spelled here for the type
                        # checker, same doctrine as the two sites above.
                        assert rendezvous is not None
                        if not rendezvous.wait_band_drained():
                            raise xr.Weg2XchgGateTimeout(
                                f"W69 Weg2XchgGateTimeout: band {bands} of "
                                f"this tag waited {rendezvous.budget_s:.0f}s "
                                f"under the #1397 band credit (Option 3) for "
                                f"its collector to drain slot {slot} of "
                                f"{slots} and none arrived. This is a CROSS "
                                f"lane (rendezvous.pair={rendezvous.pair}), "
                                f"never the diagonal, so this cannot be the "
                                f"weg2xsn30 cycle by construction -- a real "
                                f"timeout here means this lane's collector "
                                f"genuinely stalled, not that Option 3 "
                                f"cannot apply.")
                    else:
                        raise Weg2XchgBouncePhaseUnordered(
                            f"W68 Weg2XchgPlanDisagree: band {bands} of this tag "
                            f"would reuse slot {slot} of {slots} -- the assemble "
                            f"buffer does not hold the whole tag, so this deposit "
                            f"cannot complete without its collector and the "
                            f"collector cannot run before this rank's pause "
                            f"(boot weg2xsn30). Size the buffer from max_tag_bytes "
                            f"(Option 1) or reduce the pause granularity below the "
                            f"tag (Option 3, #1397: CROSS lanes only -- see "
                            f"DESIGN_option3_band_credit_0914.md); refusing "
                            f"rather than overwriting a band a consumer may "
                            f"still be reading.")
                if inflight[slot] is not None:
                    t0 = time.perf_counter()
                    ops.synchronize(c_stream)
                    collect_ms += (time.perf_counter() - t0) * 1000.0
                    inflight[slot] = None
                base = bounce.slot_address(slot)
                # ---- THE DEPOSIT HALF ------------------------------------
                #
                # Skipped entirely on a COLLECT leg: that rank does not hold
                # the source bytes (they are the peer's) and its descriptors
                # carry no `src_ptr` at all -- which is exactly why the
                # unsplit form refused it.
                if phase in (PHASE_DEPOSIT, PHASE_BOTH):
                    # #1374: NO PER-BAND CLAIM. This is where boot weg2xsn30
                    # died: the claim waited on a collector that could not run
                    # until this rank had paused, which it could not do until
                    # this loop finished. Under Option 1 the buffer holds the
                    # whole tag, so each band owns its slot and there is no
                    # reuse to protect against; the ONE wait left is
                    # `wait_drained`, once per tag, taken by the caller AFTER
                    # the credit is published. `BounceSlots._row` refuses a
                    # band beyond the table rather than folding it onto
                    # another band's row, so an undersized buffer is a named
                    # refusal instead of silent aliasing.
                    t0 = time.perf_counter()
                    if pcie_uuid:
                        from sglang.srt.managers.weg2_memory_saver import (
                            pcie_transfer_lock,
                        )
                        with pcie_transfer_lock(
                            nvml_uuid=pcie_uuid, direction=pcie_direction,
                            label=f"deposit band seq={batch.seq}",
                        ):
                            moved = _deposit_band(ops, d_stream, unit.descs,
                                                  batch, base)
                    else:
                        moved = _deposit_band(ops, d_stream, unit.descs,
                                              batch, base)
                    # The deposit MUST land before the collect reads the slot;
                    # this is the one synchronisation the pipeline cannot
                    # elide, and it is why the two halves are on two streams
                    # rather than one.  Across PROCESSES the same order is the
                    # handshake below: fill, sync, THEN post `full` -- the
                    # order `run_producer_pair` already states as its law.
                    ops.synchronize(d_stream)
                    deposit_ms += (time.perf_counter() - t0) * 1000.0
                    if moved != int(batch.total_bytes):
                        short.append(
                            f"band seq={batch.seq} deposited {moved} of "
                            f"{batch.total_bytes}")
                    deposited += moved
                    if rendezvous is not None:
                        # AFTER the sync, never before: `bytes_filled` is a
                        # post-sync claim or it is a lie the consumer acts on.
                        rendezvous.post_full(slot=slot, seq=int(batch.seq),
                                             nbytes=int(moved))
                # ---- THE COLLECT / COMPARE HALF --------------------------
                if phase in (PHASE_COLLECT, PHASE_BOTH):
                    if rendezvous is not None and phase == PHASE_COLLECT:
                        # BEFORE THE FIRST COPY, and it RAISES.  A collect that
                        # read an unposted slot would serve plausible bytes
                        # from the previous band -- the wrong layer, with every
                        # counter green.  Same rule as `run_consumer_pair`'s
                        # short-piece check, which "raises W70 and issues
                        # NOTHING".
                        if liveness is not None:
                            # #1378 xsn36/37: chunked + liveness -- a dead
                            # deposit rank dies HERE with tag and rank,
                            # instead of after a silent budget burn.
                            filled = rendezvous.wait_full_liveness(
                                slot=slot, seq=int(batch.seq),
                                liveness=liveness, tag=str(leg_name),
                                rank=int(leg_rank))
                        else:
                            filled = rendezvous.wait_full(slot=slot,
                                                          seq=int(batch.seq))
                        if filled is None:
                            # #1378 xsn42 (TEIL 1): the refusal carries its
                            # own diagnosis -- all threads + the leg state,
                            # named with the group/rank, BEFORE the raise.
                            dump_rank_stacks(
                                "W68-not-posted", tag=str(leg_name),
                                rank=int(leg_rank),
                                extra=(f"slot={slot} seq={batch.seq} "
                                       f"unit={unit.key} "
                                       f"phase={phase} mode={mode}"))
                            raise Weg2XchgBouncePhaseUnordered(
                                f"W68 Weg2XchgPlanDisagree: slot="
                                f"{slot} seq={batch.seq} was not posted full "
                                f"by any depositing rank, so this collect "
                                f"would read whatever the previous band left "
                                f"there. Issuing NOTHING.")
                        if int(filled) != int(batch.total_bytes):
                            raise Weg2XchgBouncePhaseUnordered(
                                f"W68 Weg2XchgPlanDisagree: slot="
                                f"{slot} seq={batch.seq} carries "
                                f"{int(filled)} bytes and this rank's own "
                                f"derivation of the same band is "
                                f"{int(batch.total_bytes)}. The two ends "
                                f"disagree about what is in the slot; "
                                f"issuing NOTHING rather than copying the "
                                f"overlap.")
                    if comparing:
                        # THE REFILL STAYS THE AUTHORITY.  Nothing is written
                        # to the live weights on this path -- the staged bytes
                        # are graded against what the refill already put there.
                        #
                        # NAMED HERE, NOT FIXED (out of #1385/xsn31-4's
                        # scope): `int(depth)` addresses the shadow-compare
                        # slot correctly in the PRE-OPTION-1 geometry, where
                        # `slots = assemble_slots(depth, comparing=True) =
                        # depth + 1` so index `depth` IS the one extra slot.
                        # Under Option 1 `slots = terms.lane_slots` reserves
                        # its own extra slot at index `lane_slots - 1`
                        # (`xchg_bounce.tag_slots`'s `shadow` term), which is
                        # NOT `depth` (still `terms.depth`, e.g. 2) once
                        # `slots` is 24 -- a second, latent geometry mismatch
                        # this leg's `comparing` path has not exercised on
                        # any boot yet (xsn31/4's own INJECT lines all read
                        # `verdict=NO-COMPARE pieces=0`: the flip died before
                        # a single comparison ran). Fixing the ALLOCATED SIZE
                        # (this function's whole point) does not fix this by
                        # itself; a future seat should reconcile it before
                        # trusting `mode=shadow` under a real Option-1 cap.
                        with pcie_copy_lock(pcie_uuid, pcie_direction):
                            _compare_band(ops, c_stream, unit.descs, batch,
                                          base,
                                          bounce.slot_address(int(depth)),
                                          verdict)
                    else:
                        with pcie_copy_lock(pcie_uuid, pcie_direction):
                            collected += _collect_band(ops, c_stream,
                                                       unit.descs, batch,
                                                       base)
                        inflight[slot] = batch
                    if rendezvous is not None and phase == PHASE_COLLECT:
                        ops.synchronize(c_stream)
                        inflight[slot] = None
                        # #1374: NO PER-BAND `empty` POST on row 0. That row
                        # still carries the PER-TAG drain, so posting once
                        # per band there would hand the source N drain tokens
                        # and let it overwrite the tag this collector is
                        # still reading -- the very hazard the drain exists
                        # for. The tag's single `post_drained` is the
                        # caller's, after its last band, because only the
                        # caller knows where the tag ends.
                        #
                        # #1397 OPTION 3: A DIFFERENT counter, row 1 of
                        # `full` (dead since #1374, never `empty`'s row 0
                        # above), for CROSS lanes only. MUST follow this
                        # band's own `ops.synchronize(c_stream)` on the line
                        # directly above -- posting before that sync
                        # returns is the "too early" mutant this ticket's
                        # test suite pins RED, because the depositor may
                        # legally act on this credit the instant it is
                        # visible.
                        if _band_credit_leg:
                            rendezvous.post_band_drained()
                bands += 1
        t0 = time.perf_counter()
        ops.synchronize(c_stream)
        collect_ms += (time.perf_counter() - t0) * 1000.0
    finally:
        ops.destroy_stream(d_stream)
        ops.destroy_stream(c_stream)
        # THE BUFFER IS RELEASED ON EVERY EXIT, including the raising one: the
        # user law is about host residency, and a leg that leaked its buffer on
        # the error path would hold the bytes for the life of the boot exactly
        # as the ring did.
        bounce.close()
        # #1358: THE LEG'S OWN SUMMARY, in the `finally` so a refused leg
        # reports its residue too -- a leg that raises is exactly when a slot
        # is most likely to be left held.
        try:
            log_fn = log if log is not None else logger.info
            log_fn(host_slot_leg_line(group=str(leg_group), rank=int(leg_rank),
                                      leg=str(leg_name or lane),
                                      slots_before=_slots_before))
        except BaseException:  # noqa: BLE001 -- an instrument never raises
            pass

    result = BounceResult(
        # NONE ON THE AUTHORITATIVE PATH, because nothing was graded there and
        # a NO-COMPARE verdict would read as "the grade was attempted and
        # produced nothing" rather than "no grade was asked for".
        inject=verdict if comparing else None,
        # #1336: the mode TRAVELS instead of being inferred from `inject`.
        # `mode` is the validated parameter -- unknown values were refused
        # above -- so this line and the `comparing` decision above now read
        # the same single source.
        mode=mode,
        units=len(units), bands=bands,
        deposited_bytes=deposited, collected_bytes=collected,
        planned_bytes=planned,
        host_bytes_peak=int(slot_bytes) * slots,
        slot_bytes=int(slot_bytes), depth=int(depth),
        widest_unit_key=widest.key, widest_unit_bytes=widest.nbytes,
        widest_run_bytes=widest_run(descs),
        overlap=("ok" if int(depth) >= 2 else "none"),
        deposit_ms=deposit_ms, collect_ms=collect_ms,
        short=tuple(short), banded=banded,
    )
    emit = log or logger.info
    emit("%s", result.line())
    # #1342 S3: `InjectVerdict.line()` -- the emitter of `WEG2-XCHG-INJECT`,
    # which is grading item (a) -- had NO caller anywhere, so the line had
    # never been printed by any boot at any argv.  Boot weg2xsn17 measured it
    # as 0 lines on both groups and could not tell that absence apart from
    # "the lane did not run".  THIS is the site that holds the verdict, so
    # this is the site that prints it.
    #
    # Only when the leg actually COMPARED: under `authoritative` there is no
    # comparison (`comparing = mode == INJECT_SHADOW` above, and `inject` is
    # then None), and printing a verdict for a leg that compared nothing is
    # the instrument lie #1336 closed one instance of.
    if result.inject is not None:
        emit("%s", result.inject.line())
        _INJECT_VERDICTS.append(result.inject)
        emit("%s", inject_summary_line(_INJECT_VERDICTS))
    return result




# ---------------------------------------------------------------------------
# #1330 B4n SLICE 3 -- THE PRODUCT RENDEZVOUS, over the CROSS semaphores.
# ---------------------------------------------------------------------------
#
# NO NEW SUBSTRATE.  The 24 named semaphores and the slot records they guard
# already exist and are already counted -- weg2xsn20's teardown census read 36
# = 24 CROSS (`<epoch>-<r>-<r>-<n>-{empty,full}`) + 12 diagonal -- and
# `run_producer_pair`/`run_consumer_pair` already state the order this class
# obeys: fill, ONE sync, publish (which writes `bytes_filled`), then
# `sem_post(full)`.  Building a second handshake beside them would be the
# Zweitbuchhaltung this campaign keeps paying for.
#
# ONE RENDEZVOUS PER DIRECTED CARD PAIR, because that is how the semaphores are
# named.  A leg whose descriptors span several pairs is therefore SPLIT BY PAIR
# by the caller and run once per pair; a band that mixed two pairs would post
# one pair's `full` for another pair's bytes.


#: The rendezvous wait budget for the BOOT'S LANES -- and it MUST cover the
#: deposit's STAGING: the deposit's first post lands ~123 s after the flip
#: begins (MEASURED twice: weg2xsn34's post at 123 s, weg2xsn40's W68 at the
#: 120 s expiry with the post seconds behind). The COPY itself is ~1.5 s
#: (NUTZER-Modell 14.09.); the STAGING (the deposit's pause chain) is the
#: flip's own setup phase and is what the collect's first wait has to
#: survive. History of this number: 600 s (the silent 10-minute stall,
#: NUTZER: "der timeout loest nichts, er verlaengert nur die Zeit bis zur
#: Erkennung") -> 120 s (lost the staging race twice) -> 180 s: covers the
#: measured staging with ~45 s margin, and a REAL deadlock still dies here
#: with the named W68 -- while the front's W17 gate (dc9cd96c60) keeps the
#: boot alive during the legitimate leg blocking.
LANE_RENDEZVOUS_BUDGET_S = 180.0


def dump_rank_stacks(reason: str, tag: str = "", rank: int = -1,
                     extra: str = "") -> str:
    """Dump ALL threads' Python stacks plus the rank-local leg state into
    the evidence dir, named with the reason and the rank.

    #1378 xsn42 (the coordinator's TEIL 1): the W68/W29 family died twice
    without anyone knowing what the DEPOSIT rank was doing -- the sglang
    watchdog's own thread dump (utils/watchdog.py:176, seen on weg2xsn38's
    P log) named the LanePermit deadlock only because it fired LATER than
    the leg's refusal. This dump fires AT the refusal. faulthandler is the
    cheapest writer: no external process, no ptrace, works from inside a
    C-frame (the native part is missing, the Python frames were enough at
    xsn38). The env dir is published by the launcher (build_env ->
    SGLANG_WEG2_RANKDUMP_DIR = the boot's evidence dir); without it the
    dump is skipped silently -- a rank without an evidence dir has no
    reader for it.
    """
    try:
        out_dir = os.environ.get("SGLANG_WEG2_RANKDUMP_DIR", "")
        if not out_dir:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        import faulthandler  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        ts = _time.strftime("%H%M%S")
        safe_tag = str(tag).replace("/", "_").replace(" ", "") or "notag"
        path = os.path.join(
            out_dir, f"rankdump_{reason}_{safe_tag}_r{rank}_{ts}.txt")
        with open(path, "w") as fh:
            fh.write(f"reason={reason}\ntag={tag}\nrank={rank}\n")
            if extra:
                fh.write(f"extra={extra}\n")
            fh.write("=== faulthandler.dump_traceback (all threads) ===\n")
            fh.flush()
            faulthandler.dump_traceback(file=fh)
        return path
    except BaseException:  # noqa: BLE001 -- a dump must never take the leg down
        return ""


class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def pcie_copy_lock(pcie_uuid, pcie_direction):
    """The per-copy PCIe serialisation, or a no-op when no uuid is given."""
    if not pcie_uuid:
        return _NullLock()
    from sglang.srt.managers.weg2_memory_saver import pcie_transfer_lock
    return pcie_transfer_lock(nvml_uuid=pcie_uuid,
                              direction=pcie_direction,
                              label="collect band")


#: #81 (fnFL2w3/w5/w7, 21.09.): DAS LANE-BUDGET MUSS VOR DEM FRONT-BOUND
#: GREIFEN, sonst kann es nie zuerst greifen.
#:
#: Beide Zahlen standen auf 120,0 -- diese hier und
#: `front.DRAIN_DEADLINE_DEFAULT_S` -- aus zwei verschiedenen Specs, ohne
#: voneinander zu wissen. Bei Gleichstand meldet die Front ihren STALL, waehrend
#: der Lane-Wait noch laeuft: gemessen "WEG2-FLIP STALL epoch=0 elapsed=125,2 s
#: bound=120,0 s" in w7 und 129,2 s in w3. Der Boot stirbt dann am Stall statt
#: an der benannten Lane-Verweigerung, die den Tensor und die Lane nennt --
#: dreimal hintereinander war der Stall die erste Meldung und die
#: Lane-Verweigerung nur im Traceback zu finden.
#:
#: 0,75 x 120 = 90 s laesst der Lane 25 % Vorlauf. Das ist kein Toleranzband
#: fuer langsame Deposits: gemessen deponiert die Quelle einen Chunk-Tag in
#: 643-1412 ms, also zwei Groessenordnungen darunter. Wer 90 s wartet, wartet
#: auf etwas, das nicht kommt, und soll das SAGEN.
SEQ_LANE_BUDGET_S = 90.0


class CrossSlotRendezvous:
    """The empty/full handshake for ONE directed card pair, or the diagonal.

    ``pair`` indexes :data:`weight_exchange_region.CROSS_PAIRS`; ``card`` is the
    diagonal's key instead (#1334 -- "the diagonal is one card talking to
    itself ... a pair id would be exactly the fiction that produced weg2xsn9's
    IndexError"). Exactly one of the two is given, and the SAME class serves
    both, so the diagonal is no longer the unpriced special case slice 3 left
    it as: it gets the identical short-piece check.

    THE BYTE COUNT COMES FROM THE BOUNCE'S OWN RECORD, never the region's.
    weg2xsn24 measured what sharing costs: `slot=0 seq=0 carries 1080 bytes and
    this rank's own derivation of the same band is 756323776` -- 1080 was the
    RING's number, published by `run_producer_pair`
    (`weight_exchange_transport.py:1786`) into the same (pair, slot) record.
    One ledger, two payloads. :class:`BounceSlots` is the bounce's own.

    THE ORDER IS THE LAW (`run_producer_pair`'s own words): fill, ONE sync,
    publish, THEN `sem_post(full)`. A post before the publish lets a consumer
    read the previous band's count.
    That rule is per BAND and survives #1374 word for word.

    THE CONTRACT ITSELF CHANGED IN #1374, after boot weg2xsn30 deadlocked on
    the old one. It used to be "this slot is free / this slot is filled", one
    semaphore pair per slot, and the deposit claimed a slot before filling it.
    On the co-located card that wait cannot end: D0's deposit waited on
    `empty` for PP0's collect, PP0 waited in its resume for D0's VRAM credit,
    and the credit only follows D0's pause, which only follows the deposit.
    Both budgets ran their full 120 s (D0 13:48:17->13:50:17 W68 "still full";
    PP0 13:48:13->13:50:13 W35 `peer_leg_complete=False`). Reproduced at the
    desk in two processes in seconds: test_weg2_lane_lockstep_1374.

    THE CONTRACT NOW: `full` is COUNTING per pair -- N bands are ready and the
    `seq` in the record orders them -- and the only wait is `drained`, ONCE per
    tag, taken by the source AFTER it has published that tag's credit. It
    follows from Option 1 (operator, 2026-09-13): the buffer holds a whole TAG,
    so a deposit completes without its collector, so within a tag there is
    nothing for a per-band wait to protect. The diagonal runs the identical
    contract with ``card=`` instead of ``pair=`` (#1334).
    """

    def __init__(self, sems, slots, *, pair: Optional[int] = None,
                 card: Optional[int] = None,
                 budget_s: float = SEQ_LANE_BUDGET_S):
        if (pair is None) == (card is None):
            raise ValueError(
                "exactly one of pair= (cross) or card= (diagonal) -- the two "
                "name different semaphore families and conflating them is the "
                "W15 fiction")
        self.sems = sems
        self.slots = slots
        self.pair = pair
        self.card = card
        self.budget_s = float(budget_s)

    def _slot(self, slot: int) -> int:
        return int(slot) % int(xr.SLOTS_PER_PAIR)

    def _wait(self, slot: int, kind: str, budget_s: Optional[float] = None) -> bool:
        b = self.budget_s if budget_s is None else float(budget_s)
        if self.pair is not None:
            return bool(self.sems.timedwait(self.pair, self._slot(slot), kind,
                                            b))
        return bool(self.sems.diagonal_timedwait(self.card, self._slot(slot),
                                                 kind, b))

    def _post(self, slot: int, kind: str) -> None:
        if self.pair is not None:
            self.sems.post(self.pair, self._slot(slot), kind)
        else:
            self.sems.diagonal_post(self.card, self._slot(slot), kind)

    #: The per-tag drain handshake rides row 0 of the EXISTING empty family,
    #: once per tag rather than once per band -- so the semaphore census stays
    #: 60 and the meaning of the name changes from "this slot is free" to
    #: "the previous tag has been collected".
    _DRAIN_SLOT = 0

    def prime_drain(self) -> bool:
        """Take the drain counter to 0 before the first deposit.

        `create_semaphores` arms every `empty` at 1 (region:1930) because the
        OLD contract meant "this slot is free" and the first producer had to
        pass. The NEW meaning is "the previous tag has been collected", and at
        the first tag there is no previous tag -- so an inherited 1 would let
        the source deposit tag 1 while tag 0 is still being read, and tag 1
        reuses the same band rows. Consumed non-blocking, once, by the source
        at leg start: measured in test_the_next_tag_may_not_overwrite_a_tag_
        still_being_collected, which passes WITHOUT this call for the wrong
        reason.
        """
        # weg2xsn88 (2026-09-15, buffer depth 2): DRAIN EVERY CREDIT, not
        # one. With depth d the depositor consumes only n-d of the
        # collector's n drain posts per leg, so d credits are left at a role
        # switch; a prime that took ONE left the new depositor a stale
        # credit and it ran a tag further ahead than the buffers allow --
        # measured: the p3 collect of weights_7 read a record naming
        # lm_head.weight/tag=weights (three tags later), W68 unit identity
        # mismatch, and PP0 W90 moved=66. Every leftover credit belongs to a
        # buffer the previous collector has fully drained (its group fence
        # closed that leg), so taking them all is exact, never a loss.
        n = 0
        while True:
            if self.pair is not None:
                got = bool(self.sems.trywait(self.pair, self._DRAIN_SLOT, "empty"))
            else:
                # No `diagonal_trywait` exists; a zero budget IS trywait
                # semantics (sem_timedwait with a deadline already past
                # returns ETIMEDOUT at once).
                got = bool(self.sems.diagonal_timedwait(
                    self.card, self._DRAIN_SLOT, "empty", 0.0))
            if not got:
                break
            n += 1
            if n > 64:  # a counting semaphore cannot legitimately hold this
                break
        self.last_primed = int(n)
        return n > 0

    def wait_drained(self, *, tag: str) -> bool:
        """Block until the peer has collected the PREVIOUS tag.

        THE ONE REMAINING WAIT, and it cannot deadlock: the source takes it
        AFTER it has published tag t's credit, so the collector it waits for
        is already able to run. Ordering, in the source's own sequence:
        deposit(t) -- no waits -- pause(t), credit(t), wait_drained(t),
        deposit(t+1). Without it, tag t+1 would overwrite a buffer tag t's
        collector is still reading.
        """
        return self._wait(self._DRAIN_SLOT, "empty")

    def post_drained(self, *, tag: str) -> None:
        """The collector's side: this tag is out of the buffer."""
        self._post(self._DRAIN_SLOT, "empty")

    #: The COUNTING `full` semaphore is ONE row per pair, not one per band:
    #: #1374 makes it a count ("N bands are ready") and the record -- indexed
    #: by the REAL band -- says which band each one is. Folding the band into
    #: the semaphore row is what made the first draft of this contract read
    #: band 2's record for band 0: the row held the last writer's seq.
    _COUNT_SLOT = 0

    def post_full(self, *, slot: int, seq: int, nbytes: int) -> None:
        # THE REAL BAND INDEX into the record, and the ORDER stays the law:
        # fill and sync happened in the caller, publish here, post after.
        self.slots.publish(slot=int(slot), seq=int(seq),
                           nbytes=int(nbytes), pair=self.pair, card=self.card)
        self._post(self._COUNT_SLOT, "full")

    def wait_full_liveness(self, *, slot: int, seq: int, liveness=None,
                           chunk_s: float = 2.0, tag: str = "",
                           rank: int = -1):
        """wait_full, chunked, with a LIVENESS check between the chunks.

        #1378 xsn36/37 (the co-located pair deadlock, the 5090): a wait whose
        whole budget blocks in ONE timedwait cannot tell "the peer is slow"
        from "the peer is dead" -- the W68 then names a slot that was only
        the scene of the standoff.  Chunked: every ``chunk_s`` the caller's
        liveness callback runs; a peer that is GONE dies HERE -- named with
        tag and rank, budget still running -- instead of after a silent
        hang.  ``liveness()`` answers the number of OTHER live holders of
        this boot's bounce files; zero means the deposit rank is gone.

        The budget stays the caller's (120 s, the DETECTOR per the
        coordinator's order) -- this method only makes every second of it
        auditable.
        """
        deadline = time.monotonic() + float(self.budget_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self._wait(self._COUNT_SLOT, "full", budget_s=min(chunk_s,
                                                                 remaining)):
                got_seq, nbytes = self.slots.read(slot=int(slot),
                                                  pair=self.pair,
                                                  card=self.card)
                if int(got_seq) != int(seq):
                    return None
                return int(nbytes)
            if time.monotonic() >= deadline:
                return None
            if liveness is not None and not liveness():
                raise Weg2XchgBouncePhaseUnordered(
                    f"W68 Weg2PeerGone: slot={slot} seq={seq} tag={tag!r} "
                    f"rank={rank} -- the deposit rank of this pair is no "
                    f"longer a live holder of this boot's bounce files. "
                    f"Refusing NOW with the budget still running, instead "
                    f"of naming a slot after a dead wait.")

    def wait_full(self, *, slot: int, seq: int):
        """The producer's post-sync claim, or ``None`` on a real timeout.

        THE SEQ IS CHECKED, not only the byte count: a slot posted for a
        DIFFERENT band would otherwise pass the size test whenever two bands
        happen to be equally large, which on a uniform layer stack is most of
        them. Since #1374 `full` is a COUNTING semaphore per pair -- the source
        posts once per band and the collector takes it once per band -- so the
        seq read out of the record is the ORDERING AUTHORITY, not the slot
        identity, and it is the only thing that says which band this is.
        """
        if not self._wait(self._COUNT_SLOT, "full"):
            return None
        got_seq, nbytes = self.slots.read(slot=int(slot), pair=self.pair,
                                          card=self.card)
        if int(got_seq) != int(seq):
            return None
        return int(nbytes)

    # -------------------------------------------------------------------
    # #1397 OPTION 3 -- the per-BAND drain credit, CROSS PAIRS ONLY.
    # -------------------------------------------------------------------
    #
    # NO NEW SUBSTRATE, same doctrine as `_COUNT_SLOT`/`_DRAIN_SLOT` above:
    # row 1 of the `full` family has been dead since #1374 collapsed that
    # family's meaning onto row 0 ("N bands are ready"). `full`, not
    # `empty`'s own dead row 1, on purpose: `create_semaphores` arms `full`
    # at 0 and `empty` at 1 (`SEM_ARMED_COUNTS`), so a fresh COUNTING credit
    # built on `full` starts at the value it actually wants -- no
    # `prime_drain`-style "consume the inherited one" step is needed at a
    # boot's very first tag, only `prime_band_drain`'s narrower one below.
    _BAND_DRAIN_SLOT = 1

    def _wait_nonblocking(self, slot: int, kind: str) -> bool:
        """`trywait` for the cross form; a zero-budget wait for the
        diagonal, the same substitute `prime_drain` already uses (no
        `diagonal_trywait` exists)."""
        if self.pair is not None:
            return bool(self.sems.trywait(self.pair, self._slot(slot), kind))
        return bool(self.sems.diagonal_timedwait(self.card, self._slot(slot),
                                                 kind, 0.0))

    def prime_band_drain(self) -> None:
        """Discard any credit a PRIOR tag's tail left uncollected.

        A tag's own last `min(bands, cross_lane_slots)` bands are never
        reused WITHIN that tag -- nothing ever calls `wait_band_drained`
        for them -- so up to `cross_lane_slots` posts from
        `post_band_drained` can outlive a tag's own leg. Left alone, the
        NEXT tag's first wrap would consume a credit that names no band of
        ITS OWN: a count that is right but an identity that is wrong,
        exactly the shape `wait_full`'s own seq check exists to catch on
        the OTHER counting family.

        Called ONCE, at the very start of a band-credit DEPOSIT leg, before
        this tag's own first band is deposited -- the only instant this is
        provably safe: deposit strictly leads collect within one tag
        (`post_full` for band k always precedes `post_band_drained` for
        band k), so at this instant nothing of THIS tag's own could have
        posted yet, and only a PRIOR tag's leftover credit can exist to be
        drained.
        """
        while self._wait_nonblocking(self._BAND_DRAIN_SLOT, "full"):
            pass

    def post_band_drained(self) -> None:
        """The collector's side: ONE post, AFTER this band's own sync.

        Ordering mirrors `post_full`'s own rule: this MUST follow
        `run_bounce_leg`'s `ops.synchronize(c_stream)` for this exact band,
        never precede it. A post before that synchronize returns is the
        "too early" mutant this ticket's test suite pins RED -- the
        depositor may legally overwrite this slot's host pages the instant
        it sees this credit, and the collect's own device copy may still be
        draining them.
        """
        self._post(self._BAND_DRAIN_SLOT, "full")

    def wait_band_drained(self) -> bool:
        """The depositor's side: block up to `budget_s`, or a REAL timeout.

        ONLY EVER CALLED FOR A CROSS PAIR (`self.pair is not None`) --
        `run_bounce_leg` is the enforcer, this method trusts its caller the
        same way `_wait`/`_post` already do. A diagonal lane must never
        reach here: its own collector's `resume` is gated (C14) on THIS
        SAME rank's not-yet-published VRAM credit
        (`weight_updater.py`'s pause/credit loop), so this wait would
        recreate the exact cycle #1374 deleted after boot weg2xsn30
        (`test_the_per_band_claim_cannot_return`) -- see
        `DESIGN_option3_band_credit_0914.md` for the full argument.
        """
        return self._wait(self._BAND_DRAIN_SLOT, "full")


def pair_of(src_rank: int, dst_rank: int) -> Optional[int]:
    """The CROSS_PAIRS index for a directed card pair, or ``None`` on-card.

    ``None`` is the DIAGONAL and is not an error: rank *n* of either group runs
    on ``cards[n]``, so ``src == dst`` is one card and has its own carrier.
    Handing it to ``sem_name`` is what boot weg2xsn9 reported 36 times as a
    bare IndexError, which is why that function now refuses it by name (W15).
    """
    if int(src_rank) == int(dst_rank):
        return None
    key = (int(src_rank), int(dst_rank))
    return xr.CROSS_PAIRS.index(key) if key in xr.CROSS_PAIRS else None


def _rank_of(desc: object, field: str) -> int:
    """``desc.src_rank`` / ``desc.dst_rank`` ohne Ersatzwert (#82).

    Ein Deskriptor ohne diese Felder ist kein Deskriptor, den dieser Pfad
    einordnen kann -- und ein geratener Rang schickt seine Bytes auf eine
    Lane, an der kein Partner steht.
    """
    v = getattr(desc, field, None)
    if v is None:
        name = getattr(desc, "name", None) or getattr(desc, "tag", "?")
        raise ValueError(
            f"W82 Weg2DescRankMissing: {field} is required to place "
            f"{name!r} on a lane; guessing it puts the bytes on a lane no "
            f"peer serves (fnFL2w8: P collected on c1/c2/p3 while D "
            f"deposited on c0)"
        )
    return int(v)


def group_descs_by_pair(descs: Sequence[object]):
    """``{pair_index_or_None: [descs]}`` in first-appearance order.

    ZEROFILL descriptors have no source and never cross a link; they stay with
    their destination's on-card group so the destination still memsets them.
    """
    out = {}
    for d in descs:
        if getattr(d, "kind", None) == wx.ZEROFILL:
            key = None
        else:
            # #82 (21.09.): KEINE DEFAULTS AUF DIESER ZEILE. `src_rank` und
            # `dst_rank` sind Pflichtfelder des Deskriptors
            # (weight_exchange.py:999), also kann `-1` hier nie der
            # Normalfall sein -- wohl aber der stille Ausgang, wenn jemand
            # spaeter einen anderen Deskriptor durchreicht: `pair_of(-1,-1)`
            # ist ein Paar wie jedes andere und landet in der Diagonalen,
            # wo der Aufrufer dann MEINE Karte nimmt. Drei Boots (w3/w7/w8)
            # sind an einer Diagonal-Lane gestorben, die niemand bedient;
            # dieser Default ist genau die Form, in der so etwas entsteht.
            key = pair_of(_rank_of(d, "src_rank"), _rank_of(d, "dst_rank"))
        out.setdefault(key, []).append(d)
    return out




# ---------------------------------------------------------------------------
# #1330 B4n SLICE 4 -- THE BOUNCE'S OWN SLOT RECORD.
# ---------------------------------------------------------------------------
#
# BOOT weg2xsn24 MEASURED WHY THIS CANNOT BE THE REGION'S RECORD:
#
#   W68 ... slot=0 seq=0 carries 1080 bytes and this rank's own derivation of
#   the same band is 756323776
#
# 1080 is not a corrupt number, it is SOMEONE ELSE'S. `CrossSlotRendezvous`
# published into `XchgRegion.publish` -- and `weight_exchange_transport.py:1786`
# shows `run_producer_pair` publishing into the SAME (pair, slot) records for
# the ring's own staging. Two payloads, one ledger: the collect read a count
# the ring had written for a different transfer.
#
# That is the two-bookkeepings defect, and the fix is not a bigger check but a
# record the bounce OWNS. It lives in its own shm file keyed by the boot nonce,
# beside the LayerBounce buffer both ends already map, and nothing else writes
# it.

BOUNCE_SLOT_PREFIX = "weg2-xchg-bnc-"

#: ``(seq, nbytes)`` per (pair, slot), as two int64 -- the smallest record that
#: answers "which band is in this slot and how many bytes did its producer
#: publish AFTER its own sync".
_SLOT_REC = struct.Struct("<qq")


def bounce_slots_path(boot_nonce: str, shm_root: str = "/dev/shm") -> str:
    return os.path.join(shm_root, f"{BOUNCE_SLOT_PREFIX}{boot_nonce}")


class BounceSlots:
    """The bounce's OWN (seq, bytes) table, mmapped by both ends.

    ``n_pairs`` covers the cross pairs AND the diagonal cards, so the diagonal
    stops being the unpriced special case it was in slice 3: it addresses rows
    ``N_PAIRS + card`` of the same table and therefore gets the SAME
    short-piece check as every cross pair. The COUNT_UNAVAILABLE sentinel is
    gone with it -- an instrument that could not go red on one third of the
    lanes was a gap, not a design.
    """

    def __init__(self, boot_nonce: str, *, shm_root: str = "/dev/shm",
                 create: bool = False, rows_per_pair: int = xr.SLOTS_PER_PAIR):
        # #1374 (c): THE RECORD'S ROWS ARE NOT THE SEMAPHORE CENSUS. They were
        # both `SLOTS_PER_PAIR` and `_row` folded the slot index with `%`, so a
        # lane with more slots than 2 silently aliased band 2 onto band 0's
        # row. Never bit because the #1358 default `depth=1` kept the loop on
        # slot 0 -- at `depth=2` (slots=3 under shadow) it was one boot away.
        # Option 1 (operator, 2026-09-13) sizes the buffer to a whole TAG, so
        # the row count is the BAND COUNT and comes from the terms; the
        # semaphore census stays 60 because the new contract needs one
        # counting `full` per pair, not one per band.
        self.path = bounce_slots_path(boot_nonce, shm_root)
        self.rows_per_pair = max(int(rows_per_pair), 1)
        self.rows = (xr.N_PAIRS + xr.N_CARDS) * self.rows_per_pair
        size = self.rows * _SLOT_REC.size
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if create and os.fstat(fd).st_size < size:
                os.ftruncate(fd, size)
            self._mm = _mmap.mmap(fd, size, _mmap.MAP_SHARED,
                                  _mmap.PROT_READ | _mmap.PROT_WRITE)
        finally:
            os.close(fd)

    def _row(self, pair: Optional[int], card: Optional[int], slot: int) -> int:
        """The row for one band. AN OUT-OF-RANGE SLOT RAISES; it does not wrap.

        The `%` this replaced is the whole #1374 (c) defect: a fold turns "this
        lane has more bands than the table has rows" -- a sizing error the
        terms can state -- into two bands quietly sharing one (seq, bytes)
        record, which the collector then reads as the wrong band's size.
        """
        base = int(pair) if pair is not None else xr.N_PAIRS + int(card)
        if not 0 <= int(slot) < self.rows_per_pair:
            raise Weg2XchgBouncePhaseUnordered(
                f"W68 Weg2XchgPlanDisagree: band {slot} has no row -- this "
                f"lane's record was built for {self.rows_per_pair} band(s) per "
                f"pair. The buffer and the record are sized from the SAME "
                f"terms, so a band beyond the table is a plan the two sides "
                f"do not share, not a slot to wrap onto another band's row.")
        return base * int(self.rows_per_pair) + int(slot)

    def publish(self, *, slot: int, seq: int, nbytes: int,
                pair: Optional[int] = None, card: Optional[int] = None) -> None:
        off = self._row(pair, card, slot) * _SLOT_REC.size
        self._mm[off:off + _SLOT_REC.size] = _SLOT_REC.pack(int(seq),
                                                            int(nbytes))

    def read(self, *, slot: int, pair: Optional[int] = None,
             card: Optional[int] = None) -> Tuple[int, int]:
        off = self._row(pair, card, slot) * _SLOT_REC.size
        return _SLOT_REC.unpack(self._mm[off:off + _SLOT_REC.size])

    def close(self) -> None:
        try:
            self._mm.close()
        except BaseException:  # noqa: BLE001
            pass

    def unlink(self) -> None:
        self.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# #1378 xsn44: DER SEQUENTIELLE EINHEITEN-TRANSPORT (die Layer-Form, das
# spezifizierte Design Mechanismus Punkt 2 --gewichtsaustausch-ziel-kein-
# dauer-hostram.md-- "assemble the FULL layer in a small host bounce buffer,
# then each card copies its slice"). EIN Puffer (die groesste Einheit),
# EIN full/empty-Signaalpaar je Einheit, sequenziell. Die Lane-Maschinerie
# (Lanes, Permit, Cap, Band-Grenzen, whole-leg-Locks) ist dafuer nicht
# gebaut worden und produzierte die elf Waende der Familie.
# ---------------------------------------------------------------------------

_SEQ_SLOT = 0

#: NUTZER-ORDER 2026-09-15 ("bau die beschleunigung, die sha256 nur noch als
#: option (flag), zum verifizieren, defaultmaessig off"): the per-unit CPU
#: sha256 on BOTH sides was ~50 s of CPU per 26-GB leg (weg2xsn87: a 2-GB
#: lane took ~2 s per side, ~1 GB/s, against 3-12 GB/s of PCIe). The
#: identity guard (name+tag in the deposit record) and the device-side
#: SEAM-DIGEST stay; the transport digest is a DEVELOPMENT witness now.
SEQ_UNIT_DIGEST_ENV = "SGLANG_WEG2_SEQ_UNIT_DIGEST"
#: Buffers per lane: 2 lets the depositor fill tag t+1 while the collector
#: drains tag t (the #1374 drain wait then reaches back TWO tags). 1 = the
#: xsn87 form (strict alternation, ~3 s idle per tag).
SEQ_BUFFER_DEPTH_ENV = "SGLANG_WEG2_SEQ_BUFFER_DEPTH"
#: Run a rank's lanes (c/p/p) in threads instead of one after another.
SEQ_LANES_PARALLEL_ENV = "SGLANG_WEG2_SEQ_LANES_PARALLEL"


def _env_flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default) or default).strip().lower() in (
        "1", "true", "yes", "on")


def seq_unit_digest_armed() -> bool:
    return _env_flag(SEQ_UNIT_DIGEST_ENV, "0")


def seq_buffer_depth() -> int:
    """Buffers per lane the depositor may run ahead (default 4 since
    weg2xsn99: the waker on card 1 idled 1.2 s while the depositor walked
    four of card 0's tags first; at depth 2 it could not get ahead of them).
    Cost: depth x lane bytes on tmpfs per lane (~25 GB at 4), and on-card
    staging that falls back to the host path when VRAM is short."""
    # weg2xsn284 (18.09.): DEPTH 1 IS THE DEFAULT. With registered, persistent
    # lanes (below) the flips ran 1.8-3.3 s per direction at depth 1 (vs
    # 5-12 s pageable at depth 2-4), the 5090's lanes at 12-13 GB/s; depth 4
    # parked ~23 GB of tmpfs beside the full 41 GiB arena and latched W98
    # (xsn283: shmem 58.5 GiB). The run-ahead depth 2 bought is worth less
    # than a lane at link rate.
    try:
        d = int(os.environ.get(SEQ_BUFFER_DEPTH_ENV, "1") or 1)
    except ValueError:
        d = 1
    return 1 if d < 1 else (8 if d > 8 else d)


def seq_lanes_parallel() -> bool:
    return _env_flag(SEQ_LANES_PARALLEL_ENV, "1")


def seq_lane_file_name(lane_key: str, buffer_slot: int) -> str:
    """The lane's file stem for buffer slot ``buffer_slot`` (0 = the plain
    name, so depth 1 is byte-identical to the xsn87 form)."""
    return lane_key if not buffer_slot else f"{lane_key}_s{int(buffer_slot)}"


#: NUTZER-ORDER 2026-09-15 ("bau die naechsten hebel, persistente puffer und
#: on-card per ipc"). PERSISTENT: a lane slot's host buffer is created,
#: mmapped and cudaHostRegister'ed ONCE per process (grown when a later tag
#: needs more), never unlinked per tag -- weg2xsn89 paid a register and an
#: unregister of up to 2 GB per lane per tag on both sides. The files stay
#: on tmpfs until the boot's own weg2-seq sweep. ON-CARD IPC: the on-card
#: lanes (c0/c1/c2, the two ranks of one card) stage in a cudaMalloc'ed
#: DEVICE buffer the collector opens through cudaIpcOpenMemHandle -- two D2D
#: copies instead of D2H + H2D through the host. The staging is transient
#: (allocated per tag, freed at the leg's end after every drain is confirmed,
#: see the updater), and any failure to allocate or export falls back to the
#: host path for that tag, logged.
SEQ_PERSIST_BUFFERS_ENV = "SGLANG_WEG2_SEQ_PERSIST_BUFFERS"
SEQ_ONCARD_IPC_ENV = "SGLANG_WEG2_SEQ_ONCARD_IPC"


def seq_persist_buffers() -> bool:
    return _env_flag(SEQ_PERSIST_BUFFERS_ENV, "1")


def seq_oncard_ipc() -> bool:
    return _env_flag(SEQ_ONCARD_IPC_ENV, "1")


#: Order point 2 (2026-09-15, "Transport an den Link"): ONE stream sync per
#: BATCH of units instead of one per unit. Measured on xsn122: a 1224-unit
#: cross lane of 3.5 GB spent 2.1 ms per unit in copy+sync (1.37 GB/s on an
#: 8 GB/s link), a 3776-unit lane of 12 GB reached 7.5 GB/s -- the per-unit
#: `cudaStreamSynchronize` round trip is the cost, not the link. The copies
#: of a batch are queued back to back on the lane's stream, synchronised
#: ONCE, and only then digested/recorded/posted (deposit) or placement-checked
#: (collect). The handshake stays one token per unit; the collect still waits
#: per unit before it issues that unit's copy. Env: the batch closes at
#: SGLANG_WEG2_SEQ_SYNC_BATCH_MIB (default 64) or at
#: SGLANG_WEG2_SEQ_SYNC_BATCH_UNITS (default 32) units, whichever first;
#: UNITS=1 restores the per-unit form.
SEQ_SYNC_BATCH_MIB_ENV = "SGLANG_WEG2_SEQ_SYNC_BATCH_MIB"
SEQ_SYNC_BATCH_UNITS_ENV = "SGLANG_WEG2_SEQ_SYNC_BATCH_UNITS"


def seq_sync_batch() -> tuple:
    """``(max_bytes, max_units)`` of one sync batch; clamped to sane values."""
    try:
        mib = float(os.environ.get(SEQ_SYNC_BATCH_MIB_ENV, "64"))
    except ValueError:
        mib = 64.0
    try:
        units = int(os.environ.get(SEQ_SYNC_BATCH_UNITS_ENV, "32"))
    except ValueError:
        units = 32
    return int(max(1.0, mib) * (1 << 20)), max(1, min(units, 4096))


def _piece_verbose(i: int, n: int) -> bool:
    """Order point 2: the per-piece deposit/cf/collect lines were ~10k log
    lines per flip in the scheduler processes. Keep the first four, every
    64th and the last piece of a lane call -- the SIGSEGV hunt the cf line
    served (xsn57-61) reads 'died at i or i+1' off the highest index that
    ARRIVED, which sampling still gives to within 64. Refusals, NO-WRITE,
    lane-time, first-piece-done and ptrattr stay unconditional."""
    return i < 4 or i % 64 == 0 or i == n - 1


def _sync_groups(pieces, max_bytes: int, max_units: int) -> list:
    """Index groups of ``pieces`` that share one synchronize. Deterministic
    from the piece list and the two bounds alone (both sides derive it, but
    only the sync cadence depends on it -- the handshake is per unit)."""
    groups, cur, cur_bytes = [], [], 0
    for i, piece in enumerate(pieces):
        nb = int(piece.nbytes)
        if cur and (len(cur) >= max_units or cur_bytes + nb > max_bytes):
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(i)
        cur_bytes += nb
    if cur:
        groups.append(cur)
    return groups


_SEQ_HOST_BUF: dict = {}   # path -> {"fh","mm","addr","size","registered"}
_SEQ_STAGE: dict = {}      # (boot_nonce, lane_file) -> {"ptr","size","device","ops"}
_SEQ_CACHE_LOCK = __import__("threading").Lock()


def _persistent_host_buffer(path: str, biggest: int, ops, lane_key: str, log):
    """``(mm, addr, registered_str, refusal_str)`` for one lane slot, cached
    per process; grown (unregister, remap, re-register) when ``biggest``
    exceeds the cached size."""
    with _SEQ_CACHE_LOCK:
        ent = _SEQ_HOST_BUF.get(path)
        if ent is not None and int(ent["size"]) >= int(biggest):
            log(f"WEG2-SEQ persist lane={lane_key} reuse size={int(ent['size'])} "
                f"registered={ent['registered']}")
            return ent["mm"], ent["addr"], ent["registered"], ""
        how = "new"
        if ent is not None:
            how = f"grow {int(ent['size'])}->{int(biggest)}"
            if ent["registered"] == "yes":
                try:
                    ops.host_unregister(int(ent["addr"]))
                except Exception:  # noqa: BLE001
                    pass
            try:
                ent["mm"].close()
                ent["fh"].close()
            except Exception:  # noqa: BLE001
                pass
            _SEQ_HOST_BUF.pop(path, None)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size < biggest:
            with open(path, "ab") as _tfh:
                _tfh.truncate(biggest)
        fh = open(path, "r+b")
        mm = _mmap.mmap(fh.fileno(), biggest)
        addr = _mmap_addr(mm)
        # xsn262 (17.09.): cudaHostRegister over freshly truncated tmpfs
        # pages faults every page in under the driver's lock -- 553 MB
        # took 21 s with four processes registering at once (they
        # serialise in the driver). Populate the mapping first, outside
        # any lock, so the register only pins. MADV_POPULATE_WRITE (Linux
        # 5.14+, value 23); a kernel without it leaves the old form.
        _t_pop = time.perf_counter()
        try:
            mm.madvise(getattr(_mmap, "MADV_POPULATE_WRITE", 23))
            _pop_ms = (time.perf_counter() - _t_pop) * 1000
        except (OSError, ValueError, AttributeError):
            _pop_ms = -1.0
        registered = "no"
        refusal = ""
        t0 = time.perf_counter()
        try:
            # xsn268 (17.09., py-spy --native at the DRAIN-STALL): PP0 and TP0
            # of one card sat in cudaHostUnregister (ioctl) for the whole
            # ~12.5 s stall -- the lane release / the grow path unregisters a
            # 2 GB mapping while the co-located process copies, and the
            # driver serialises both. Registering became cheap with the
            # populate above; UNregistering did not. Default: no
            # registration at all -- the lane copies run pageable (the driver
            # stages them through its own pinned buffers) and the release is
            # an unmap + truncate with no driver call. SGLANG_WEG2_SEQ_HOST_
            # REGISTER=1 keeps the pinned form for an A/B.
            if not seq_host_register():
                registered = "no(off)"
            else:
                ops.host_register(int(addr), int(biggest), tp.CUDA_HOST_REGISTER_PORTABLE)
                registered = "yes"
        except AttributeError:
            registered = "unavailable"
        except Exception as _reg_exc:  # noqa: BLE001
            registered = f"no({type(_reg_exc).__name__})"
            refusal = (f"host_register failed for lane {lane_key} addr={int(addr)} "
                       f"bytes={int(biggest)}: {type(_reg_exc).__name__}: {_reg_exc}")
        log(f"WEG2-SEQ persist lane={lane_key} {how} size={int(biggest)} "
            f"addr={int(addr)} registered={registered} "
            f"register_ms={(time.perf_counter() - t0) * 1000:.0f} "
            f"populate_ms={_pop_ms:.0f}")
        if refusal:
            mm.close()
            fh.close()
            return None, 0, registered, refusal
        _SEQ_HOST_BUF[path] = {"fh": fh, "mm": mm, "addr": int(addr),
                               "size": int(biggest), "registered": registered,
                               "ops": ops, "lane": lane_key}
        return mm, int(addr), registered, ""


#: xsn265 (17.09.): the PERSISTENT lane buffers grew, over both flip
#: directions, to ~25 GB of tmpfs against 15.75 GiB priced -- and under the
#: DFLASH form (draft arena beside the KV arena) the host ledger latched W98
#: (cushion 1.42 < 1.50 GiB, shmem 61 GiB) 18 s after the first flip. What
#: made the buffers persistent was the register cost (weg2xsn89: 2 GB per
#: lane per tag); with the tmpfs populate before cudaHostRegister that cost
#: is 22-146 ms per lane, so the buffers are now released at every LEG END:
#: the depositor (after `_weg2_xchg_drain_outstanding`'s drain waits, i.e.
#: after the collector confirmed every band) unregisters, unmaps and
#: TRUNCATES the file to 0 -- tmpfs residency returns to the host -- and the
#: collector unregisters and unmaps its own mapping. The next leg creates,
#: populates and registers again. =0 keeps the boot-long form.
SEQ_RELEASE_LANES_ENV = "SGLANG_WEG2_SEQ_RELEASE_LANES"
#: xsn268: cudaHostRegister/Unregister of the host lane buffers -- OFF by
#: default (see `_persistent_host_buffer`): the unregister was the 12.5-s
#: stall measured on every flip.
SEQ_HOST_REGISTER_ENV = "SGLANG_WEG2_SEQ_HOST_REGISTER"


def seq_host_register() -> bool:
    """weg2xsn284 (18.09.): REGISTERED BY DEFAULT again -- once per lane,
    never unregistered (see seq_release_lanes): the 12.5-s stall of xsn265/
    267 was the per-leg cudaHostUnregister, not the register. Measured with
    register=1/release=0/depth=1: 5090 lanes 12-13 GB/s (pageable 1.6-3.9),
    card 2 7-12 GB/s, the x4 slot 2.7-6 GB/s; flips 1.8-3.3 s per direction."""
    return _env_flag(SEQ_HOST_REGISTER_ENV, "1")


def seq_release_lanes() -> bool:
    """weg2xsn284: lanes stay mapped, populated and registered across legs
    (default OFF = no release). The residency this keeps is depth x lane
    bytes (~6 GB at depth 1) and is what the host ledger prices."""
    return _env_flag(SEQ_RELEASE_LANES_ENV, "0")


def release_host_lane_buffers(*, truncate: bool, log=None) -> Tuple[int, int]:
    """Unregister and unmap every cached lane host buffer of this process;
    with ``truncate`` the files are cut to 0 bytes (the depositor's side,
    after every band was drained). Returns ``(buffers, bytes)``."""
    emit = log or logger.info
    n = 0
    total = 0
    t0 = time.perf_counter()
    with _SEQ_CACHE_LOCK:
        for path, ent in list(_SEQ_HOST_BUF.items()):
            try:
                if ent.get("registered") == "yes" and ent.get("ops") is not None:
                    try:
                        ent["ops"].host_unregister(int(ent["addr"]))
                    except Exception as exc:  # noqa: BLE001 -- unmapped below regardless
                        emit(f"WEG2-SEQ lane-release lane={ent.get('lane')} "
                             f"host_unregister failed: {type(exc).__name__}: {exc}")
                try:
                    ent["mm"].close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    ent["fh"].close()
                except Exception:  # noqa: BLE001
                    pass
                if truncate:
                    try:
                        os.truncate(path, 0)
                    except OSError as exc:
                        emit(f"WEG2-SEQ lane-release lane={ent.get('lane')} "
                             f"truncate failed: {exc}")
                n += 1
                total += int(ent.get("size", 0) or 0)
            finally:
                _SEQ_HOST_BUF.pop(path, None)
    emit(f"WEG2-SEQ lane-release buffers={n} bytes={total} truncated={int(bool(truncate))} "
         f"ms={(time.perf_counter() - t0) * 1000:.0f} -- host lane buffers of this leg "
         f"returned (tmpfs residency falls with the truncate; the next leg registers "
         f"again, populate+register measured 22-146 ms per lane)")
    return n, total


def _stage_free_entry(ent, log, lane_key: str) -> None:
    """Free one staging entry and give its bytes back to whoever charged
    them (``ent['refund']``, set by :func:`_stage_alloc` from ``charge``)."""
    try:
        ent["ops"].raw_free(int(ent["ptr"]))
    except Exception as exc:  # noqa: BLE001
        log(f"WEG2-SEQ stage lane={lane_key} free-failed: {exc}")
    refund = ent.get("refund")
    if refund is not None:
        try:
            refund()
        except Exception as exc:  # noqa: BLE001
            log(f"WEG2-SEQ stage lane={lane_key} refund-failed: {exc}")


def _stage_alloc(ops, device: int, key, nbytes: int, log, lane_key: str,
                 charge=None) -> int:
    """A transient DEVICE staging buffer for one lane slot; an existing
    entry of the same slot is freed first (the caller runs only after that
    slot's drain). Freed for good by :func:`release_stage_buffers`.

    ``charge`` (weg2xsn269, 18.09.): ``(nbytes) -> refund-callable | None``.
    The staging sits on the card the WAKING rank resumes into, so its bytes
    must be booked against that card's VRAM credit BEFORE the cudaMalloc --
    unbooked, the waker's credit check and this allocation raced (xsn269:
    TP0 checked 3124 MiB free for a 2502 MiB tag, PP0 staged 1.99 GB 240 ms
    later, TP0's cuMemCreate ran out of memory and the memory saver's
    exit(1) killed the rank). A ``None`` from ``charge`` refuses the staging
    (host path for the tag); the refund runs when the entry is freed."""
    with _SEQ_CACHE_LOCK:
        # weg2xsn100 (depth 4): at most TWO live stagings per lane -- the
        # depositor's run-ahead otherwise parks 4 x 2 GB on the waker's
        # card, the waker's credit wait starves and W108 fires. Slots
        # beyond the cap take the host path (the caller's fallback).
        _live = [k for k in _SEQ_STAGE
                 if k[0] == key[0] and k != key
                 and str(k[1]).split("_s")[0] == str(key[1]).split("_s")[0]]
        if len(_live) >= 2:
            raise RuntimeError(f"staging cap: {len(_live)} live stagings on lane "
                               f"{lane_key} (max 2)")
        ent = _SEQ_STAGE.pop(key, None)
        if ent is not None:
            _stage_free_entry(ent, log, lane_key)
        refund = None
        if charge is not None:
            refund = charge(int(nbytes))
            if refund is None:
                raise RuntimeError(
                    f"staging refused by the card's VRAM credit: {int(nbytes)} "
                    f"B on lane {lane_key} exceed the balance the waking rank "
                    f"was not promised")
        try:
            ptr = int(ops.raw_malloc(int(device), int(nbytes)))
        except Exception:
            if refund is not None:
                refund()
            raise
        _SEQ_STAGE[key] = {"ptr": ptr, "size": int(nbytes), "device": int(device),
                           "ops": ops, "refund": refund}
        return ptr


def release_stage_buffers(boot_nonce=None, log=None) -> int:
    """Free every staging buffer this process holds (``boot_nonce`` None =
    all) -- the depositor's leg end, after every drain is confirmed."""
    n = 0
    _log = log if log is not None else (lambda *_a: None)
    with _SEQ_CACHE_LOCK:
        for key in [k for k in _SEQ_STAGE
                    if boot_nonce is None or k[0] == str(boot_nonce)]:
            ent = _SEQ_STAGE.pop(key)
            try:
                ent["ops"].raw_free(int(ent["ptr"]))
                n += 1
            except Exception as exc:  # noqa: BLE001
                _log(f"WEG2-SEQ stage {key[1]} free-failed: {exc}")
            refund = ent.get("refund")
            if refund is not None:
                try:
                    refund()
                except Exception as exc:  # noqa: BLE001
                    _log(f"WEG2-SEQ stage {key[1]} refund-failed: {exc}")
    if log is not None and n:
        log(f"WEG2-SEQ stage released={n}")
    return n


def sequential_buffer_path(boot_nonce: str, shm_root: str = xr.SHM_ROOT,
                           lane: str = "") -> str:
    # #1378 xsn53: PER LANE.  Three cards run their co-located pairs at the
    # same time; a boot-wide file would put all three pairs' bytes on the
    # same windows.  The lane key is the handshake's own (p<pair> / c<card>).
    return (f"{shm_root}/weg2-seq-{boot_nonce}/"
            + (f"{lane}_" if lane else "") + "unit_buffer.bin")


def sequential_digest_path(boot_nonce: str, shm_root: str = xr.SHM_ROOT,
                           lane: str = "") -> str:
    return (f"{shm_root}/weg2-seq-{boot_nonce}/"
            + (f"{lane}_" if lane else "") + "unit_digests.json")


def _mmap_addr(mmv: "_mmap.mmap") -> int:
    """The mmap's host virtual address, for ops.host_register."""
    import ctypes
    return ctypes.addressof(ctypes.c_char.from_buffer(mmv))


def _mm_slice(mm, start: int, length: int) -> memoryview:
    """A memoryview of the mmap's region -- for the desk digest_fn."""
    return memoryview(mm)[start:start + length]


def ptr_attrs(addr: int) -> Tuple[int, int, int]:
    """``(rc, type, device)`` for one address, as the DRIVER sees it.

    ONE producer for this reading, used by the transport's copy-out probe AND by
    the wake path's post-resume probe -- two implementations of the same
    measurement is the defect class this ticket has paid for repeatedly.

    ``type`` is ``cudaMemoryType``: 0 unregistered, 1 host, 2 device, 3 managed.
    A ``type`` of 0 with ``device`` -1 and ``rc`` 0 is the decisive reading: the
    call SUCCEEDED and the driver does not know the address -- which is what a
    reserved-but-not-committed VMM range looks like, and what weg2xsn61 found
    under the destination of the first copy-out (dst_rc=0 dst_type=0
    dst_device=-1, src_type=1) two seconds before the SIGSEGV.

    THIS IS A READ AND CANNOT FAULT: ``cudaPointerGetAttributes`` returns
    ``cudaErrorInvalidValue`` for an unknown address instead of dereferencing
    it, so the probe never adds a crash to the path it measures. Resolved from
    the ALREADY-LOADED runtime (torch has libcudart in process) rather than by
    dlopen'ing a second copy; on any failure it reports ``(-1, -1, -1)`` rather
    than raising, because a diagnostic that can break its caller is worse than
    none.
    """
    # #1378 xsn67/xsn68 -- THE PROBE KILLED THE RANK IT MEASURED. The first
    # version built the ctypes Structure CLASS and a fresh `CDLL(None)` handle
    # on EVERY call; one call per tag (xsn66) survived, the purity census
    # (hundreds of calls per tag, xsn67/xsn68) died inside the ffi call with
    # faulthandler's "Garbage-collecting" on the very frame -- SIGSEGV on P
    # rank 0 with no log line, read for two boots as "the resume hangs". The
    # type object, the function handle and its argtypes are now built ONCE
    # and held for the process's life; nothing here is created per call.
    #
    # #1378 xsn69 -- AND THE SECOND WAY THE PROBE KILLED ITS RANK: with the
    # binding cached, all three P ranks still died within two seconds of the
    # first resume, each in a DIFFERENT pure-Python or C frame (tag_slots
    # arithmetic, json.raw_decode, torch.cuda.memory_stats on the corridor
    # thread) -- heap corruption, and P rank 1/2 had made exactly ONE call.
    # MEASURED on this box: `CDLL(None).cudaPointerGetAttributes` resolves into
    # nvidia/cu13/lib/libcudart.so.13 while torch runs on libcudart.so.12 --
    # both are mapped in the process, and the first call into the cu13 runtime
    # initialises a SECOND runtime's state over the cu12 context.  D, which
    # never ran the probe, never crashed.  The reading now comes from the
    # DRIVER API (libcuda.so.1, ONE library, the one torch_memory_saver itself
    # maps pages with): cuPointerGetAttribute(MEMORY_TYPE) and (DEVICE_ORDINAL).
    # Same contract: (rc, type, device); driver memory types are HOST=1
    # DEVICE=2 ARRAY=3 UNIFIED=4, an unknown address is rc!=0 and reported as
    # type 0 / device -1, which is what the decisive reading has always been.
    try:
        fn = _ptr_attrs_binding()
        mem_type, rc = _cu_pointer_attr_int(fn, 2, addr)   # CU_POINTER_ATTRIBUTE_MEMORY_TYPE
        if rc != 0:
            return int(rc), 0, -1
        dev, rc2 = _cu_pointer_attr_int(fn, 9, addr)       # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        return 0, int(mem_type), (int(dev) if rc2 == 0 else -1)
    except Exception:  # noqa: BLE001 -- see the docstring's last paragraph
        return -1, -1, -1


_PTR_ATTRS_BINDING = None


def _ptr_attrs_binding():
    """``cuPointerGetAttribute`` from libcuda.so.1, bound ONCE per process."""
    global _PTR_ATTRS_BINDING
    if _PTR_ATTRS_BINDING is None:
        import ctypes as _ct

        lib = _ct.CDLL("libcuda.so.1")
        fn = lib.cuPointerGetAttribute
        fn.restype = _ct.c_int
        # (void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
        fn.argtypes = [_ct.c_void_p, _ct.c_int, _ct.c_ulonglong]
        # the CDLL object is kept alive by the tuple: a function pointer whose
        # library handle was collected is the same defect one layer down
        _PTR_ATTRS_BINDING = (fn, lib)
    return _PTR_ATTRS_BINDING[0]


def _cu_pointer_attr_int(fn, attribute: int, addr: int) -> Tuple[int, int]:
    """``(value, rc)`` of one integer-valued pointer attribute."""
    import ctypes as _ct

    out = _ct.c_uint(0)
    rc = fn(_ct.byref(out), int(attribute), _ct.c_ulonglong(int(addr)))
    return int(out.value), int(rc)


def _mmap_addr(mm: "_mmap.mmap") -> int:
    """The mmap's host virtual address, for ops.memcpy_async and
    ops.host_register. Same conversion the LayerBounce uses at :623-627:
    ctypes.addressof(c_char.from_buffer(mmap)) -- an INTEGER, not a
    memoryview. The xsn46/xsn47 wall was a memoryview passed where the
    original call site expects an int address (the fake ops was shaped
    to accept the view, not to match the real receiver)."""
    import ctypes
    holder = ctypes.c_char.from_buffer(mm)
    addr = ctypes.addressof(holder)
    del holder
    return addr


def run_sequential_units(descs, ops, boot_nonce: str, *,
                         slot_bytes: int,
                         shm_root: str = xr.SHM_ROOT,
                         device: int = 0,
                         phase: str = PHASE_DEPOSIT,
                         #: weg2xsn269: the depositor's on-card IPC staging is
                         #: booked against the card's VRAM credit through this
                         #: ``(nbytes) -> refund | None`` before it is
                         #: allocated; None = unbooked (desk fakes, ring arm).
                         stage_charge=None,
                         #: #1378 xsn44 (the PLACEMENT witness): the digest of
                         #: the DESTINATION after the copy-out, compared
 #: against the deposit's record BY NAME. The sha256 buffer digest
 #: witnesses the TRANSPORT (the right bytes read from the buffer); this
 #: witness covers the PLACEMENT (the bytes landed at the right address) --
 #: the swapped-destination mutant passes the buffer digest green-falsely
 #: and dies here.  Optional: the desk tests provide it; the metal uses the
 #: SEAM-DIGEST machinery (the position-weighted fold, 362856ea7c).
                         dst_digest_fn=None,
                         #: weg2xsn86 (#1378): ``{(tag, param_name)}`` this
                         #: rank must CONSUME but not WRITE -- a MEASURED
                         #: target share (the draft's embed_tokens on D is
                         #: the target's own tensor, set_embed_and_head), whose
                         #: bytes the target's own leg carries and whose VA is
                         #: still paused while the draft tag is collected
                         #: (SIGSEGV in cuMemcpyAsync at draft unit 2, TP0/1/2).
                         #: The unit stays in the lane so the two sides' unit
                         #: lists keep their indices: the source (PP2) holds
                         #: its OWN copy and deposits it.
                         no_write=None,
                         #: buffer slot of this lane for this tag (0/1 under
                         #: depth 2) -- BOTH sides derive it from the same
                         #: per-lane tag counter, see the updater.
                         buffer_slot: int = 0,
                         #: None = the env flag (SEQ_UNIT_DIGEST_ENV, default
                         #: off); True/False forces it (desk witnesses).
                         unit_digest=None,
                         liveness=None,
                         budget_s: float = SEQ_LANE_BUDGET_S,
                         #: #1378 xsn44: the shared buffer as a bytearray.
                         #: When provided, the transport uses it directly
                         #: (bytearray slices for the copies) -- no mmap,
                         #: no pointer conversion, no pinning. The desk
                         #: tests pass a bytearray; the metal path creates
                         #: a mmap and passes its memoryview.
                         buffer=None,
                         #: #1378 xsn53 (DIE IDENTITAET): the lane's own
                         #: handshake key -- EXACTLY ONE of the two, the same
                         #: contract :class:`CrossSlotRendezvous` states.  A
                         #: CROSS lane names its directed card pair; the
                         #: DIAGONAL names its own card (rank n of either
                         #: group runs on cards[n], ``pair_of``'s law).  The
                         #: xsn52 wall was this call site asking the cross
                         #: path with the literal ``pair=0`` for an on-card
                         #: lane; the xsn53 desk finding was it then passing
                         #: the UNIT INDEX as the diagonal's card, which
                         #: bounds out at the fourth unit.  Neither id may be
                         #: guessed here: the caller that grouped the lanes
                         #: knows which one this is.
                         pair: Optional[int] = None,
                         card: Optional[int] = None,
                         log=None) -> str:
    """Der sequenzielle Einheiten-Transport: EIN Puffer, zwei Signale je
    Einheit, der Digest je Einheit als Bedingung.

    ``units``: die Einheiten-Liste -- je Einheit (name, tag, nbytes,
    src_addr, dst_addr), IDENTISCH auf beiden Seiten (die PLAN-PARAM-
    Liste; gemessen: beide Seiten planen alle Tags).
    ``phase``: PHASE_DEPOSIT (kopiere src->Puffer, sync, Digest ueber den
    Puffer, schreibe den Digest, post full) oder PHASE_COLLECT (warte
    full [chunked, liveness-gekoppelt], lies den Digest des Deposits,
    vergleiche mit dem Digest ueber den Puffer -- BY NAME, Mismatch =
    Refusal mit beiden Digests --, kopiere Puffer->dst, sync, post
    consumed).

    Der Puffer: EIN shm-File je LANE, gross genug fuer die Summe der
    Einheiten, von beiden Seiten der Lane gemappt. JE LANE und nicht je
    Boot: drei Karten fahren ihre Paare GLEICHZEITIG, ein bootweites File
    wuerde die drei Paare auf dieselben Bytes legen. Je Einheit EIN
    Offset-Fenster (der laufende Summe), weil der Deposit ohne Wartung des
    Collectors weiterschreibt -- die #1374-Vertragsform ("full is COUNTING",
    der einzige Wait ist der je Tag, den der Aufrufer hinter dieser Funktion
    fuehrt).  Kein host_register (die xsn47-Lektion: die Doppelregistrierung
    kam aus der Ueberlappung mit dem LayerBounce-Bereich, nicht aus dem
    Mapping selbst).

    Das Signal: das full der Lane auf _SEQ_SLOT, je Einheit ein Token. Das
    empty dieser Zeile ist der je-Tag-Drain des Aufrufers
    (``CrossSlotRendezvous._DRAIN_SLOT == 0``) -- diese Funktion postet es
    NICHT, sonst gaebe sie dem Drain-Zaehler Token, die den Deposit des
    naechsten Tags freigeben wuerden, bevor der Collect sie gelesen hat
    (gemessen als Weg2XchgBouncePhaseUnordered-Familie).

    Der Digest: ``digest_fn(bytes) -> hex`` ueber die Puffer-Region; der
    Deposit schreibt seinen Digest in die Digest-Datei, der Collect
    vergleicht seinen eigenen (ueber dieselben Shared-Memory-Bytes)
    dagegen. Der Zeuge fuer das MAPPING liegt in der Zuordnung
    unit->(src,dst): der Verschiebungs-Mutant (die Einheit i mapped die
    Bytes der Einheit i+1) produziert andere Puffer-Inhalte -- der
    Digest des Collects ueber die falsch gemappte dst-Adresse weicht vom
    Deposit-Digest ab und die Refusal nennt beide.

    ``liveness``: der Co-Card-Check zwischen den Wait-Chunks (die
    xsn36/37-Lehre: eine Wait ohne Aliveness-Pruefung verwandelt den
    Deadlock in Schweigen). Fail-open.
    """
    import hashlib  # noqa: PLC0415
    import json as _json  # noqa: PLC0415
    import time as _time  # noqa: PLC0415

    # #1378 xsn53: THE LANE'S OWN HANDSHAKE, RESOLVED AND LOGGED ONCE PER
    # SIDE.  The coordinator's cheapest proof that the two ends of a lane
    # meet BY NAME: this line is emitted by the deposit rank and by the
    # collect rank, and the two must spell the same semaphore.  It is also
    # the guard: a caller that cannot say which lane it holds is refused
    # here rather than resolved onto somebody else's semaphore.
    if (pair is None) == (card is None):
        raise ValueError(
            "run_sequential_units: exactly one of pair= (cross) or card= "
            "(diagonal) must be given -- the two name different semaphore "
            "families and a caller that cannot say which lane it holds is "
            "the xsn52 shape (all six ranks on one foreign cross name)")
    log = log or (lambda *a: None)
    if pair is not None:
        lane_key = f"p{int(pair)}"
        # THE CROSS PATH'S NAME COMES FROM THE TWO CARDS, so a caller that
        # holds an on-card lane and asks the cross path is refused here by
        # name (W15) instead of resolving a foreign pair's semaphore -- the
        # exact shape that hung xsn52 on all six ranks at once.
        _src, _dst = xr.CROSS_PAIRS[int(pair)]
        resolved_full = xr.cross_sem_name(boot_nonce, _src, _dst, _SEQ_SLOT,
                                          "full")
        _is_diagonal = False
    else:
        lane_key = f"c{int(card)}"
        resolved_full = xr.diagonal_sem_name(boot_nonce, int(card), _SEQ_SLOT,
                                             "full")
        _is_diagonal = True
    # #1378 xsn56: `slot_bytes` ON THE LINE, because it became a per-lane
    # magnitude rather than a boot-wide constant.  The caller now hands the
    # lane's OWN byte sum (operator order 2026-09-15), so this is the number a
    # reader needs to price /dev/shm against -- and the one that made
    # weg2xsn55's "3 batches" refusal readable only from a traceback.
    # #82 (21.09.): DER ERSTE TENSOR UND SEIN RANGPAAR AUF DIE ZEILE.
    # Beim Vergleich der beiden Seiten von fnFL2w8 liess sich aus dem Log
    # ablesen, WELCHE Lanes jede Seite bediente (D: 6, P: 9) -- aber nicht,
    # WARUM eine Lane entstand. Die Antwort steckt im Rangpaar des ersten
    # Deskriptors, und genau die musste ich aus Manifesten rekonstruieren,
    # die der Teardown danach geraeumt hatte.
    _d0 = descs[0] if descs else None
    _who = (f" first={getattr(_d0, 'param_name', None) or getattr(_d0, 'tag', '?')!r}"
            f" src_rank={getattr(_d0, 'src_rank', '?')}"
            f" dst_rank={getattr(_d0, 'dst_rank', '?')}") if _d0 is not None else ""
    log(f"WEG2-SEQ lane={lane_key} phase={phase} handshake={resolved_full} "
        f"descs={len(descs)} slot={_SEQ_SLOT} slot_bytes={int(slot_bytes)}"
        f"{_who}")

    # THE PIECES, DERIVED IDENTICALLY ON BOTH SIDES.  ``batch_descs`` is the
    # one producer of the slot layout the lane form already used: deterministic
    # from the descriptor list alone, FLAT and STRIDED2D handled with the same
    # pitch arithmetic the lane's own loops run, and a shape it cannot price
    # REFUSED rather than copied as padding.  Feeding it the whole lane's bytes
    # as one slot yields ONE batch whose ``slot_off`` fields are the running
    # offsets this form stages the pieces at -- the W90 fix and the offset
    # arithmetic from the same tested code, not two derivations of one fact.
    _batches = tp.batch_descs(list(descs), slot_bytes=int(slot_bytes),
                              first_seq=0)
    if len(_batches) != 1:
        raise ValueError(
            f"run_sequential_units: the lane's bytes do not fit one buffer "
            f"(slot_bytes={int(slot_bytes)} produced {len(_batches)} batches) "
            f"-- the caller's slot_bytes is the lane's priced size and a "
            f"second batch would need a second handshake this form does not "
            f"have")
    _batch = _batches[0]
    total_bytes = int(_batch.total_bytes)
    biggest = total_bytes
    if biggest <= 0:
        return "no units"
    # #1378 xsn53: PER LANE, not per boot -- three cards run their pairs at
    # the same time and a boot-wide file would put all three pairs' bytes on
    # top of each other.
    _digest_on = (seq_unit_digest_armed() if unit_digest is None
                  else bool(unit_digest))
    _lane_file = seq_lane_file_name(lane_key, int(buffer_slot or 0))
    path = sequential_buffer_path(boot_nonce, shm_root, lane=_lane_file)
    dpath = sequential_digest_path(boot_nonce, shm_root, lane=_lane_file)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    _persist = False
    if buffer is not None:
        # the desk path: the test provides the buffer directly
        buf = buffer
        _owns_buf = False
        _seq_mm = None
        _fh = None
    elif seq_persist_buffers():
        buf, _p_addr, _p_reg, _p_refusal = _persistent_host_buffer(
            path, int(biggest), ops, lane_key, log)
        if _p_refusal:
            return _p_refusal
        _owns_buf = False          # cached: never unregistered/closed/unlinked here
        _seq_mm = buf
        _fh = None
        _persist = True
    else:
        # #1378 xsn53 (DER PUFFER IST WIEDER GETEILT): the SHARED tmpfs mmap,
        # per lane, NO host_register.  69f727dac2 had replaced this with
        # ``ctypes.create_string_buffer`` -- a PRIVATE per-process allocation
        # (its own comment said "per-process, not shared"), so the collect
        # read its own zero-filled buffer and the transport could not move a
        # byte between the two ranks.  The mapping itself was never the rc=712
        # problem: the xsn47 lesson is that the OVERLAP came from
        # cudaHostRegister on a range the LayerBounce had already registered,
        # so the shared mmap comes back WITHOUT the registration.
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size < biggest:
            with open(path, "wb") as _tfh:
                _tfh.truncate(biggest)
        _fh = open(path, "r+b")
        buf = _mmap.mmap(_fh.fileno(), biggest)
        _owns_buf = True
        _seq_mm = buf
        # #1378 xsn60: THE REGISTRATION COMES BACK, TOLERANTLY -- the comment
        # above already names why removing it outright was the wrong repair:
        # "the OVERLAP came from cudaHostRegister on a range the LayerBounce had
        # already registered". The answer to an overlap is to TOLERATE the
        # already-registered case, exactly as the LayerBounce does at :629-634
        # ("a 7.5 % regression, not a refusal"), not to drop the pin for every
        # range including the ones nobody else holds.
        #
        # WHY IT IS THE SUSPECT, measured rather than assumed: the DEPOSIT
        # direction (D2H, device -> this buffer) moves bytes -- 144 pieces with
        # sha256 on weg2xsn55. The COLLECT direction (H2D, this buffer ->
        # device) takes a SIGSEGV inside the very first `memcpy_async`
        # (weg2xsn59: one `cf i=0/114` row at 05:52:16, dead at 05:52:19, no row
        # for i=1). Same function, opposite direction, and the pin is what the
        # H2D path needs that the D2H path does not.
        #
        # The outcome is LOGGED either way, because "registered" and "tolerated
        # an overlap" and "failed for another reason" are three different states
        # and a silent pin would make the next boot guess again.
        # The address is taken OUTSIDE the try on purpose: Python resolves
        # `ops.host_register` BEFORE it evaluates the arguments, so a fake
        # without that method raises AttributeError first and an inline
        # assignment would never run -- which is exactly how the desk suite
        # caught this on the first execution (UnboundLocalError, 8 failures).
    if buffer is None:
        _seq_addr = _mmap_addr(buf)
        _seq_registered = "no"
        _reg_refusal = ""
        try:
            if _persist:
                _seq_registered = str(_p_reg)
            else:
                ops.host_register(int(_seq_addr), int(biggest),
                                  tp.CUDA_HOST_REGISTER_PORTABLE)
                _seq_registered = "yes"
        except AttributeError:
            # the desk fakes carry no pin at all: unpinned, and nothing to
            # unregister later
            _seq_registered = "unavailable"
        except Exception as _reg_exc:  # noqa: BLE001
            # #1378 xsn70 -- A FAILED PIN IS A REFUSAL, NOT A DEGRADE. The
            # tolerant form ("no(RuntimeError)", carry on unpinned) hid the
            # defect this commit fixes: the previous lane's buffer was closed
            # WITHOUT cudaHostUnregister, so the driver still held its VA
            # range pinned to the OLD pages; the next lane's mmap landed on the
            # same range, its register failed (already registered), and the
            # copy engine read the stale registration -- p2's 72 pieces
            # "matched" on the CPU digest while the DMA source was another
            # file's pages, and p4's last piece, 1804 bytes past the stale
            # range, died with cudaMemcpyAsync rc=1 invalid argument.
            _seq_registered = f"no({type(_reg_exc).__name__})"
            _reg_refusal = (f"host_register failed for lane {lane_key} "
                            f"addr={int(_seq_addr)} bytes={int(biggest)}: "
                            f"{type(_reg_exc).__name__}: {_reg_exc} -- a copy "
                            f"out of an unpinnable range would read whatever "
                            f"registration the driver still holds there")
        log(f"WEG2-SEQ register lane={lane_key} bytes={int(biggest)} "
            f"addr={int(_seq_addr)} registered={_seq_registered}")
        if _reg_refusal:
            if _fh is not None:
                _seq_mm.close()
                _fh.close()
            return _reg_refusal
    try:
        sems = tp.SemSet(boot_nonce)
        # The lane the lane-form ran its copies on, or the default stream when the
        # device ops do not expose stream creation (the desk fakes).
        try:
            stream = ops.create_stream(device)
        except Exception:  # noqa: BLE001 -- a copy stream is an optimisation
            stream = 0
        base_addr = _mmap_addr(buf) if _seq_mm is not None else 0
        _ipc_base, _ipc_hex, _ipc_opened = 0, "", False
        _dep_recs = {}   # weg2xsn91: this tag's records, kept in memory
        _t_wait = _t_copy = _t_rec = 0.0   # weg2xsn92: where a lane's time goes
        _t_lane0 = time.perf_counter()
        if (phase == PHASE_DEPOSIT and card is not None and seq_oncard_ipc()
                and hasattr(ops, "ipc_get_handle") and hasattr(ops, "raw_malloc")):
            try:
                _ipc_base = _stage_alloc(ops, int(device),
                                         (str(boot_nonce), _lane_file),
                                         int(total_bytes), log, lane_key,
                                         charge=stage_charge)
                _ipc_hex = bytes(ops.ipc_get_handle(int(_ipc_base))).hex()
                log(f"WEG2-SEQ ipc lane={lane_key} phase=deposit stage={int(_ipc_base)} "
                    f"bytes={int(total_bytes)} handle={_ipc_hex[:16]}.. -- on-card "
                    f"D2D staging, the collector opens it by IPC")
            except Exception as _ipc_exc:  # noqa: BLE001
                log(f"WEG2-SEQ ipc lane={lane_key} phase=deposit UNAVAILABLE "
                    f"({type(_ipc_exc).__name__}: {_ipc_exc}) -- host path for "
                    f"this tag")
                _ipc_base, _ipc_hex = 0, ""
        # #1378 xsn55 (WO DER DEPOSIT STEHT): xsn55's diagonal lane blocked with
        # NO c0 buffer file and NO c0 digest json, while both cross lanes wrote
        # buffers and digests immediately -- so the block was between the lane
        # line above and the first piece's digest write.  Two lines make that
        # interval readable on the metal instead of inferred: one after the
        # mapping (with the lane's byte size), one after the FIRST piece's copy
        # + digest.  Everything between them is the piece loop's own body.
        log(f"WEG2-SEQ mapped lane={lane_key} phase={phase} bytes={total_bytes} "
            f"units={len(_batch.pieces)} path={path}")
        # Order point 2: batched syncs (see `seq_sync_batch`). The unit loop
        # keeps every witness of the per-unit form -- identity by name, the
        # transport digest, NO-WRITE, the placement digest, one `full` per
        # unit -- and moves only the `synchronize` to the batch boundary.
        # DEPOSIT: copies of a batch queued, ONE sync, then digest+record+post
        # per unit (the post is what licenses the collect to read the window,
        # so it can only follow the sync). COLLECT: per unit wait+record+
        # identity+digest, copy queued; ONE sync per batch; then the placement
        # witness per unit (it reads the destination, so only after the sync).
        _bat_bytes, _bat_units = seq_sync_batch()
        _groups = _sync_groups(_batch.pieces, _bat_bytes, _bat_units)
        _n_sync = 0
        _first_done = False
        for _grp in _groups:
            if phase == PHASE_DEPOSIT:
                _issued = []
                _tc0 = time.perf_counter()
                for i in _grp:
                    piece = _batch.pieces[i]
                    desc = descs[piece.desc_index]
                    window_off = int(piece.slot_off)
                    name = getattr(desc, "param_name", "?")
                    tag = getattr(desc, "tag", "")
                    nbytes = int(piece.nbytes)
                    if desc.src_ptr is None:
                        return (f"deposit at piece {i} {name!r}: the desc carries no "
                                f"src_ptr -- this rank does not hold the bytes this "
                                f"lane says it moves")
                    src_ptr = int(desc.src_ptr) + int(piece.src_off)
                    # the D2H (or D2D staging) copy into this piece's window,
                    # the SAME pitch arithmetic the lane form runs
                    _buf_addr = (_ipc_base + window_off) if _ipc_base else (base_addr + window_off)
                    if piece.kind == tp.FLAT:
                        ops.memcpy_async(_buf_addr, src_ptr, nbytes, stream)
                    else:
                        ops.memcpy2d_async(_buf_addr, int(piece.run_bytes), src_ptr,
                                           int(piece.spitch), int(piece.run_bytes),
                                           int(piece.rows), stream)
                    _issued.append((i, window_off, name, tag, nbytes))
                ops.synchronize(stream)
                _n_sync += 1
                _t_copy += time.perf_counter() - _tc0
                for (i, window_off, name, tag, nbytes) in _issued:
                    t0 = _time.perf_counter()
                    digest = (hashlib.sha256(
                        bytes(buf[window_off:window_off + nbytes])).hexdigest()[:16]
                        if (_digest_on and not _ipc_base) else "")
                    recs = _dep_recs
                    recs[str(i)] = {"name": name, "tag": tag, "digest": digest, "ipc": _ipc_hex,
                                    "nbytes": nbytes}
                    # weg2xsn90/xsn111: ONE SMALL FILE PER UNIT, atomic
                    # tmp+rename -- the collector reads it after the `full`.
                    _tr0 = time.perf_counter()
                    _upath = f"{dpath}.u{i}"
                    _tmp = _upath + ".tmp"
                    with open(_tmp, "w") as fh:
                        _json.dump(recs[str(i)], fh)
                    os.replace(_tmp, _upath)
                    _t_rec += time.perf_counter() - _tr0
                    # #1378 xsn53: THE LANE'S OWN TOKEN, one full per unit on
                    # the lane's resolved handshake -- after the sync above.
                    if _is_diagonal:
                        sems.diagonal_post(int(card), _SEQ_SLOT, "full")
                    else:
                        sems.post(int(pair), _SEQ_SLOT, "full")
                    if _piece_verbose(i, len(_batch.pieces)):
                        log(f"WEG2-SEQ deposit piece {i} {name!r} tag={tag!r} "
                            f"nbytes={nbytes} window={window_off} digest={digest} "
                            f"ms={(_time.perf_counter()-t0)*1000:.1f}")
                    if not _first_done:
                        _first_done = True
                        # #1378 xsn55: the SECOND marker of the mapping interval.
                        log(f"WEG2-SEQ first-piece-done lane={lane_key} phase={phase} "
                            f"ms={(_time.perf_counter()-_t_lane0)*1000:.1f}")
                continue
            # ---- COLLECT ----
            _issued = []
            _dst_probe, _probed = tp.dst_pointer_probe(ops), set()
            for i in _grp:
                piece = _batch.pieces[i]
                desc = descs[piece.desc_index]
                window_off = int(piece.slot_off)
                name = getattr(desc, "param_name", "?")
                tag = getattr(desc, "tag", "")
                nbytes = int(piece.nbytes)
                label = (f"piece {i} {name!r} tag={tag!r} nbytes={nbytes} "
                         f"window={window_off}")
                _tw0 = time.perf_counter()
                deadline = _time.monotonic() + float(budget_s)
                got = False
                while _time.monotonic() < deadline:
                    # weg2xsn91: BLOCK on the semaphore in 0.2 s slices
                    if _is_diagonal:
                        _got = sems.diagonal_timedwait(int(card), _SEQ_SLOT,
                                                       "full", 0.2)
                    else:
                        _got = sems.timedwait(int(pair), _SEQ_SLOT, "full", 0.2)
                    if _got:
                        got = True
                        break
                    if liveness is not None and not liveness():
                        dump_rank_stacks(
                            "PeerGone-seq", tag=str(name), rank=int(window_off),
                            extra=f"unit {i} {name!r} -- the deposit peer is "
                                  f"gone while the collect waited")
                        return f"PeerGone at unit {i} {name!r}"
                if not got:
                    dump_rank_stacks(
                        "budget-expired-seq", tag=str(name), rank=int(window_off),
                        extra=f"unit {i} {name!r} budget={budget_s}s")
                    return f"budget expired at unit {i} {name!r}"
                _t_wait += time.perf_counter() - _tw0
                # weg2xsn90: the deposit posts `full` only AFTER its record is
                # written (atomically) -- wait for it, never proceed on `{}`.
                dep = {}
                _t_rec0 = time.perf_counter()
                while True:
                    dep = {}
                    _upath = f"{dpath}.u{i}"
                    if os.path.exists(_upath):
                        try:
                            with open(_upath) as _rf:
                                dep = _json.load(_rf) or {}
                        except ValueError:
                            dep = {}
                    if dep:
                        break
                    if time.perf_counter() - _t_rec0 > 5.0:
                        return (f"record missing for unit {i} {name!r} on lane "
                                f"{lane_key} after 5 s -- the deposit posts "
                                f"'full' only after writing it")
                    time.sleep(0.002)
                _t_rec += time.perf_counter() - _t_rec0
                if dep.get("ipc") and not _ipc_base:
                    # the deposit staged on-card: open its handle once
                    _ipc_base = int(ops.ipc_open_handle(bytes.fromhex(str(dep["ipc"]))))
                    _ipc_opened = True
                    log(f"WEG2-SEQ ipc lane={lane_key} phase=collect opened={int(_ipc_base)} "
                        f"handle={str(dep['ipc'])[:16]}..")
                # weg2xsn84 (#1378): THE RECORD NAMES THE TENSOR, SO CHECK IT --
                # a unit whose record names another tensor is the pause/resume
                # orders diverging, refused BEFORE the copy-out.
                _dep_name = str(dep.get("name", "") or "")
                _dep_tag = str(dep.get("tag", "") or "")
                _my_tag = str(getattr(desc, "tag", "") or "")
                if dep and (_dep_name != str(name)
                            or (_dep_tag and _my_tag and _dep_tag != _my_tag)):
                    dump_rank_stacks(
                        "unit-identity-mismatch-seq", tag=str(name),
                        rank=int(window_off),
                        extra=f"unit {i} deposit={_dep_name!r}/{_dep_tag} "
                              f"collect={name!r}/{_my_tag}")
                    return (f"unit identity mismatch at unit {i}: the deposit "
                            f"record names {_dep_name!r} tag={_dep_tag or '-'}, "
                            f"this collect expects {name!r} tag="
                            f"{_my_tag or '-'} -- the source's pause order and "
                            f"this rank's resume order diverged (one lane "
                            f"buffer, per-tag lockstep #1374); refused before "
                            f"the copy-out")
                dep_digest = str(dep.get("digest", ""))
                my_digest = (hashlib.sha256(
                    bytes(buf[window_off:window_off + nbytes])).hexdigest()[:16]
                    if dep_digest else "off")
                if dep_digest and my_digest != dep_digest:
                    dump_rank_stacks(
                        "digest-mismatch-seq", tag=str(name), rank=int(window_off),
                        extra=f"unit {i} {name!r} deposit={dep_digest} "
                              f"collect={my_digest}")
                    return (f"digest mismatch at unit {i} {name!r}: "
                            f"deposit={dep_digest} collect={my_digest}")
                if no_write and (str(getattr(desc, "tag", "") or ""),
                                 str(name)) in no_write:
                    # weg2xsn86: consumed, NOT written -- see `no_write`.
                    log(f"WEG2-SEQ collect piece {i} {name!r} "
                        f"tag={getattr(desc, 'tag', '')!r} digest={my_digest} "
                        f"matches deposit NO-WRITE: a MEASURED target share on "
                        f"this rank, the target's own leg carries these bytes")
                    continue
                if desc.dst_ptr is None:
                    return (f"collect at piece {i} {name!r}: the desc carries no "
                            f"dst_ptr -- this rank does not hold the destination "
                            f"this lane says it fills")
                dst_ptr = int(desc.dst_ptr) + int(piece.dst_off)
                _buf_addr = (_ipc_base + window_off) if _ipc_base else (base_addr + window_off)
                # #1378 xsn57/xsn59: the numbers BEFORE the copy-out, every
                # piece, so a traceback-less SIGSEGV still names its piece.
                if _piece_verbose(i, len(_batch.pieces)):
                    log(f"WEG2-SEQ cf lane={lane_key} i={i}/{len(_batch.pieces)} "
                        f"dst={int(dst_ptr)} src={int(_buf_addr)} base={int(base_addr)} "
                        f"off={window_off} n={nbytes} k={piece.kind} nm={name!r}")
                if i == 0:
                    # #1378 xsn61: ask the driver whether it knows the address
                    _drc, _dty, _ddev = ptr_attrs(int(dst_ptr))
                    _src, _sty, _sdev = ptr_attrs(int(_buf_addr))
                    log(f"WEG2-SEQ ptrattr lane={lane_key} "
                        f"dst_rc={_drc} dst_type={_dty} dst_device={_ddev} "
                        f"src_rc={_src} src_type={_sty} src_device={_sdev}")
                # fnFL2x34: unit 2 (the draft's lm_head, 636 MB) went into the
                # PAUSED target head -- SIGSEGV two lines after the i==0 probe
                # above said type=2 for unit 0. Every destination, once.
                _why = tp.refuse_unmapped_dst(_dst_probe, _probed, dst=int(desc.dst_ptr),
                                              lane_key=lane_key, i=i, name=name, tag=tag)
                if _why:
                    return _why
                _tc0 = time.perf_counter()
                if piece.kind == tp.FLAT:
                    ops.memcpy_async(int(dst_ptr), _buf_addr, nbytes, stream)
                else:
                    # on the way OUT a STRIDED2D piece scatters (spitch = run)
                    ops.memcpy2d_async(int(dst_ptr), int(piece.dpitch), _buf_addr,
                                       int(piece.run_bytes), int(piece.run_bytes),
                                       int(piece.rows), stream)
                _t_copy += time.perf_counter() - _tc0
                _issued.append((i, name, int(dst_ptr), nbytes, dep_digest, my_digest, label))
            if _issued:
                _tc0 = time.perf_counter()
                ops.synchronize(stream)
                _n_sync += 1
                _t_copy += time.perf_counter() - _tc0
            for (i, name, dst_ptr, nbytes, dep_digest, my_digest, label) in _issued:
                if dst_digest_fn is not None:
                    # #1378 xsn44 (the PLACEMENT witness): what LANDED at this
                    # destination vs what the deposit recorded -- after the sync.
                    dst_digest = dst_digest_fn(int(dst_ptr), nbytes)
                    if dep_digest and dst_digest != dep_digest:
                        dump_rank_stacks(
                            "placement-mismatch-seq", tag=str(name),
                            rank=int(window_off),
                            extra=f"unit {i} {name!r} deposit={dep_digest} "
                                  f"destination={dst_digest} -- the bytes "
                                  f"landed at the wrong place (the swapped-"
                                  f"destination shape, silent until quality)")
                        return (f"placement mismatch at unit {i} {name!r}: "
                                f"deposit={dep_digest} destination={dst_digest}")
                # NO `empty` POST HERE -- the per-tag DRAIN is the caller's
                # (`CrossSlotRendezvous.prime_drain/wait_drained/post_drained`).
                if _piece_verbose(i, len(_batch.pieces)):
                    log(f"WEG2-SEQ collect {label} digest={my_digest} "
                        f"matches deposit")
        log(f"WEG2-SEQ lane-time lane={lane_key} phase={phase} units={len(_batch.pieces)} "
            f"bytes={total_bytes} total_ms={(time.perf_counter() - _t_lane0) * 1000:.0f} "
            f"wait_ms={_t_wait * 1000:.0f} copy_sync_ms={_t_copy * 1000:.0f} "
            f"record_ms={_t_rec * 1000:.0f} ipc={'yes' if _ipc_base else 'no'} "
            f"syncs={_n_sync} batch={_bat_units}u/{_bat_bytes >> 20}MiB "
            f"t={_time.time():.3f} t0={_time.time() - (time.perf_counter() - _t_lane0):.3f}")
        if phase == PHASE_COLLECT and _owns_buf and _seq_mm is not None:
            # #1385's lesson, at this form's own site: the file is freed by the
            # side that reads it LAST (the collect; the deposit's next tag is
            # gated on this side's drained post).  Unlinking keeps the pages
            # alive for this process's own mapping and drops them at close.
            try:
                os.unlink(path)
            except OSError:
                pass
        return ""
    finally:
        if _ipc_opened:
            try:
                ops.ipc_close_handle(int(_ipc_base))
            except Exception as _cl_exc:  # noqa: BLE001
                log(f"WEG2-SEQ ipc lane={lane_key} close-failed: {_cl_exc}")
        if _owns_buf and _seq_mm is not None:
            # #1378 xsn70: UNREGISTER BEFORE THE MUNMAP, or the driver keeps this
            # VA range pinned to these pages after they are gone -- the next
            # lane's mmap reuses the range, its own register fails, and its
            # copies read the stale pin (see the register site above).
            if _seq_registered == "yes":
                try:
                    ops.host_unregister(int(_seq_addr))
                    log(f"WEG2-SEQ unregister lane={lane_key} addr={int(_seq_addr)} ok")
                except Exception as _unreg_exc:  # noqa: BLE001
                    log(f"WEG2-SEQ unregister lane={lane_key} addr={int(_seq_addr)} "
                        f"FAILED {type(_unreg_exc).__name__}: {_unreg_exc} -- the "
                        f"range stays pinned; the next lane on it will refuse")
            _seq_mm.close()
        if _owns_buf and _fh is not None:
            _fh.close()
