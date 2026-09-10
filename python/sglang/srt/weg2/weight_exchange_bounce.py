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
* **path (a), byte-identical pieces** -- where the source's piece and the
  destination's piece are the same bytes, the existing cross-pair staging lane
  moves them card-to-card (``weight_exchange_transport.run_leg``).  This module
  adds no transport for that case; :func:`agreed_descs` only says which
  descriptors qualify, and the AUTHORITY question is settled by handing the
  transport the ORIGINAL descriptors instead of the shadow's re-targeted ones.
  Measured on boot weg2xsn8 this set is 4.90 MiB against a 27.52 GiB image, so
  it is real and byte-negligible.  It is not over-built here for that reason.
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
    "agreed_descs",
    "bounce_path",
    "leg_geometry",
    "plan_units",
    "refuse_if_plan_exceeds_slot",
    "refuse_if_slot_short",
    "refuse_if_staging_disagrees",
    "registered_bounce_bytes",
    "run_agreed_leg",
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


def refuse_if_staging_disagrees(terms: xb.BounceTerms) -> None:
    """Refuse when the CHARGED staging total is not the lane's real allocation.

    ONE LAUNCHER VALUE MUST FEED BOTH GEOMETRIES (operator order 2026-09-10;
    section 10.8 step 3's "one reader, or they drift", recorded in 10.10).
    ``xr.DATA_BYTES`` is what the staging lane actually IS -- the region file is
    ``DATA_OFF + DATA_BYTES``, and ``XchgRegion.open`` refuses unless the
    header's geometry matches this build's constants
    (``weight_exchange_region.py:659``), so the module constant and the shared
    file cannot disagree.  ``terms.staging_bytes`` is what the #1269 ledger was
    CHARGED.  A difference means the boot was priced against a lane that does
    not exist -- in either direction -- and the reap bound is wrong by exactly
    that difference.

    IT GRADES THE TOTAL AND NOT THE WIDTH, and that choice is measured rather
    than lazy (10.10).  Today the region cuts its lane as ``N_PAIRS 6 x
    SLOTS_PER_PAIR 2 x SLOT_BYTES 32 MiB`` while the launcher charges ``pairs 3
    x 2 x 64 MiB`` -- both 384 MiB.  The charge is therefore correct while the
    two decompositions disagree about the slot, so grading the WIDTH here would
    refuse a boot whose ledger is right, for a labelling defect; and warning
    and continuing would be the #505a class.  The width is pinned instead by
    ``test_the_totals_agreeing_does_not_prove_the_widths_agree``, a CANARY that
    goes red the moment section 10.8 step 3's 32 -> 64 raise lands without the
    launcher's inputs following it -- at which point the total becomes 768
    against 384 and THIS refusal fires too.

    W68 and no new code: ``Weg2XchgPlanDisagree``'s own docstring is "two
    things that must agree about this exchange do not", which is this exactly.
    It is the REGION's class, because that is the one the transport raises
    (there are two classes of that name -- see 10.10).
    """
    charged = int(terms.staging_bytes)
    actual = int(xr.DATA_BYTES)
    if charged == actual:
        return
    raise xr.Weg2XchgPlanDisagree(
        f"W68 Weg2XchgPlanDisagree staging: the ledger was charged "
        f"{charged} B of path-(a) staging (pairs={terms.pairs} x "
        f"slots_per_pair={xb.SLOTS_PER_PAIR} x slot_bytes={terms.slot_bytes}) "
        f"but the lane that exists is {actual} B (xr.N_PAIRS={xr.N_PAIRS} x "
        f"xr.SLOTS_PER_PAIR={xr.SLOTS_PER_PAIR} x "
        f"xr.SLOT_BYTES={xr.SLOT_BYTES}). One launcher value must feed the "
        f"region header and bounce_terms(slot_bytes=) -- see section 10.8 "
        f"step 3 and 10.10. The boot is otherwise priced against a lane that "
        f"does not exist and the reap bound is wrong by "
        f"{abs(charged - actual)} B."
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
    return int(terms.buffer_bytes) // int(terms.depth), int(terms.depth)


# ---------------------------------------------------------------------------
# The carrier.
# ---------------------------------------------------------------------------


def bounce_path(boot_nonce: str, shm_root: str = xr.SHM_ROOT) -> str:
    """The group-wide assemble buffer's file, one per boot.

    ONE FILE FOR THE GROUP, not one per card like
    ``tp.oncard_host_path``: the point of path (b) is that a unit is assembled
    ONCE and every destination reads its own rows out of that one copy, so a
    per-card file would be the image again, three times over.  It lives in the
    same boot directory as the region and joins the same residue sweep.
    """
    return os.path.join(xr.region_dir(boot_nonce, shm_root), "bounce.bin")


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
                 shm_root: str = xr.SHM_ROOT) -> None:
        if int(slot_bytes) <= 0:
            raise ValueError(f"slot_bytes must be positive, not {slot_bytes!r}")
        if int(depth) <= 0:
            raise ValueError(f"depth must be positive, not {depth!r}")
        self.ops = ops
        self.slot_bytes = int(slot_bytes)
        self.depth = int(depth)
        self.nbytes = self.slot_bytes * self.depth
        self.path = bounce_path(boot_nonce, shm_root)
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


def agreed_descs(descs: Sequence[object],
                 agreed_names: Sequence[str]) -> List[object]:
    """The descriptors path (a) may carry: those whose parameter both ends
    agreed byte-identically.

    A FILTER AND NOT A TRANSPORT.  The bytes move through the existing cross
    lane; what makes them AUTHORITATIVE rather than observed is that the
    caller hands ``tp.run_leg`` these descriptors with their ORIGINAL
    ``dst_ptr`` instead of the shadow's ``sh.to_shadow`` re-targeting.  The
    agreement itself is the manifest reconciliation's answer
    (``sh.reconcile_card_manifest``) and is not recomputed here -- one
    producer of that verdict.
    """
    want = {str(n) for n in agreed_names}
    return [d for d in descs if str(getattr(d, "param_name", "")) in want]


# ---------------------------------------------------------------------------
# The result, and what its verdict does and does not claim.
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
            f"banded={len(self.banded)}"
            + (f" banded_units={','.join(self.banded)}" if self.banded else "")
            + f" overlap={self.overlap} verdict={self.verdict}"
        )


@dataclass
class AgreedResult:
    """Path (a)'s accounting.  Same verdict semantics as :class:`BounceResult`."""

    pieces: int
    moved_bytes: int
    planned_bytes: int
    slot_bytes: int

    @property
    def verdict(self) -> str:
        return "MATCH" if self.moved_bytes == self.planned_bytes else "SHORT"

    def line(self) -> str:
        return (
            "WEG2-XCHG-AGREED "
            f"pieces={self.pieces} moved={self.moved_bytes} "
            f"planned={self.planned_bytes} slot_bytes={self.slot_bytes} "
            f"verdict={self.verdict}"
        )


# ---------------------------------------------------------------------------
# Path (b): the streaming loop.
# ---------------------------------------------------------------------------


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


def _missing_pointer(descs: Sequence[object]) -> Optional[str]:
    """W74's case: a destination whose slice no source covers.

    Assembly does not CREATE a source, it only stages one, so a descriptor
    with no ``src_ptr`` on the depositing side (or no ``dst_ptr`` on the
    collecting side) is a hole in the plan and must be named, never zeroed.
    """
    for d in descs:
        if getattr(d, "src_ptr", None) is None:
            return f"{getattr(d, 'param_name', '?')} has no source pointer"
        if getattr(d, "dst_ptr", None) is None:
            return f"{getattr(d, 'param_name', '?')} has no destination pointer"
    return None


def run_bounce_leg(
    descs: Sequence[object],
    ops: tp.DeviceOps,
    boot_nonce: str,
    *,
    slot_bytes: Optional[int] = None,
    depth: Optional[int] = None,
    terms: Optional[xb.BounceTerms] = None,
    shm_root: str = xr.SHM_ROOT,
    device: int = 0,
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
    if terms is not None:
        derived_slot, derived_depth = leg_geometry(terms)
        slot_bytes = derived_slot if slot_bytes is None else slot_bytes
        depth = derived_depth if depth is None else depth
    if slot_bytes is None or depth is None:
        raise ValueError(
            "run_bounce_leg needs either `terms` (the ARM's priced decision) "
            "or an explicit slot_bytes/depth pair; it derives no size of its "
            "own, because the sizing expression has one owner (xchg_bounce)"
        )
    descs = list(descs)
    hole = _missing_pointer(descs)
    if hole is not None:
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing bounce: {hole} -- assembly stages a "
            f"source, it does not create one, so this slice would be served "
            f"with undefined bytes. Refusing before the buffer is mapped."
        )
    refuse_if_slot_short(slot_bytes, descs)
    # AMENDMENT 2: a unit is assembled COMPLETE in one depth-slot.  This is the
    # run-moment half of the ARM's coverage grade -- see the function's
    # docstring for why one check cannot cover both moments.
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

    bounce = LayerBounce(ops, boot_nonce, slot_bytes=slot_bytes, depth=depth,
                         create=True, shm_root=shm_root)
    d_stream = ops.create_stream(device)
    c_stream = ops.create_stream(device)
    #: What each slot is still draining, so a slot is never re-deposited under
    #: a collect that has not landed.  This list IS the pipeline's state.
    inflight: List[Optional[object]] = [None] * int(depth)
    deposited = collected = 0
    bands = 0
    deposit_ms = collect_ms = 0.0
    short: List[str] = []
    try:
        for unit in units:
            for batch in tp.batch_descs(list(unit.descs), int(slot_bytes)):
                slot = bands % int(depth)
                if inflight[slot] is not None:
                    t0 = time.perf_counter()
                    ops.synchronize(c_stream)
                    collect_ms += (time.perf_counter() - t0) * 1000.0
                    inflight[slot] = None
                base = bounce.slot_address(slot)
                t0 = time.perf_counter()
                moved = _deposit_band(ops, d_stream, unit.descs, batch, base)
                # The deposit MUST land before the collect reads the slot; this
                # is the one synchronisation the pipeline cannot elide, and it
                # is why the two halves are on two streams rather than one.
                ops.synchronize(d_stream)
                deposit_ms += (time.perf_counter() - t0) * 1000.0
                if moved != int(batch.total_bytes):
                    short.append(
                        f"band seq={batch.seq} deposited {moved} of "
                        f"{batch.total_bytes}")
                deposited += moved
                collected += _collect_band(ops, c_stream, unit.descs, batch,
                                           base)
                inflight[slot] = batch
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

    result = BounceResult(
        units=len(units), bands=bands,
        deposited_bytes=deposited, collected_bytes=collected,
        planned_bytes=planned,
        host_bytes_peak=int(slot_bytes) * int(depth),
        slot_bytes=int(slot_bytes), depth=int(depth),
        widest_unit_key=widest.key, widest_unit_bytes=widest.nbytes,
        widest_run_bytes=widest_run(descs),
        overlap=("ok" if int(depth) >= 2 else "none"),
        deposit_ms=deposit_ms, collect_ms=collect_ms,
        short=tuple(short), banded=banded,
    )
    (log or logger.info)("%s", result.line())
    return result


def run_agreed_leg(
    descs: Sequence[object],
    ops: tp.DeviceOps,
    boot_nonce: str,
    *,
    slot_bytes: Optional[int] = None,
    shm_root: str = xr.SHM_ROOT,
    device: int = 0,
    log=None,
) -> AgreedResult:
    """Path (a), AUTHORITATIVE: byte-identical pieces to their LIVE storage.

    The transport is the existing staging lane's own arithmetic and the slot is
    ``xr.SLOT_BYTES``; the single thing that makes this the authority rather
    than an observer is that the descriptors keep their ORIGINAL ``dst_ptr``.
    The shadow's ``sh.to_shadow`` exists precisely to take that away, and the
    absence of that call here is the whole of B2.

    It is deliberately NOT built on top of ``tp.run_leg``: that function drives
    six directed pairs through the shared region's semaphores and the wave
    gate, which is the right shape for a cross-process flip and the wrong one
    for a set measured at 4.90 MiB.  The staging buffer is the same
    :class:`LayerBounce` carrier at the lane's own slot size, so path (a) adds
    no second host term of its own.
    """
    slot = int(xr.SLOT_BYTES if slot_bytes is None else slot_bytes)
    descs = list(descs)
    hole = _missing_pointer(descs)
    if hole is not None:
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing agreed: {hole} -- an agreed piece "
            f"whose storage one end does not hold is not agreed."
        )
    refuse_if_slot_short(slot, descs)
    planned = sum(int(getattr(d, "nbytes", 0)) for d in descs)
    bounce = LayerBounce(ops, boot_nonce, slot_bytes=slot, depth=1,
                         create=True, shm_root=shm_root)
    stream = ops.create_stream(device)
    moved = 0
    pieces = 0
    try:
        for batch in tp.batch_descs(descs, slot):
            base = bounce.slot_address(0)
            _deposit_band(ops, stream, descs, batch, base)
            ops.synchronize(stream)
            moved += _collect_band(ops, stream, descs, batch, base)
            ops.synchronize(stream)
            pieces += len(batch.pieces)
    finally:
        ops.destroy_stream(stream)
        bounce.close()
    result = AgreedResult(pieces=pieces, moved_bytes=moved,
                          planned_bytes=planned, slot_bytes=slot)
    (log or logger.info)("%s", result.line())
    return result
