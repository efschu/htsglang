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
                    moved = _deposit_band(ops, d_stream, unit.descs, batch,
                                          base)
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
                        filled = rendezvous.wait_full(slot=slot,
                                                      seq=int(batch.seq))
                        if filled is None:
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
                        _compare_band(ops, c_stream, unit.descs, batch, base,
                                      bounce.slot_address(int(depth)), verdict)
                    else:
                        collected += _collect_band(ops, c_stream, unit.descs,
                                                   batch, base)
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
                 card: Optional[int] = None, budget_s: float = 120.0):
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

    def _wait(self, slot: int, kind: str) -> bool:
        if self.pair is not None:
            return bool(self.sems.timedwait(self.pair, self._slot(slot), kind,
                                            self.budget_s))
        return bool(self.sems.diagonal_timedwait(self.card, self._slot(slot),
                                                 kind, self.budget_s))

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
        if self.pair is not None:
            return bool(self.sems.trywait(self.pair, self._DRAIN_SLOT, "empty"))
        # No `diagonal_trywait` exists; a zero budget IS trywait semantics
        # (sem_timedwait with a deadline already past returns ETIMEDOUT at
        # once), and that keeps the diagonal on the identical contract without
        # widening SemSet for one caller.
        return bool(self.sems.diagonal_timedwait(self.card, self._DRAIN_SLOT,
                                                 "empty", 0.0))

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
            key = pair_of(getattr(d, "src_rank", -1), getattr(d, "dst_rank", -1))
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
