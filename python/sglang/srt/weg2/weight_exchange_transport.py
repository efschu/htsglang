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
"""#1273 slice S4 -- the transport: cross-card staging and the on-card lane.

WEG2_REUSE_SPEC_0908 section 6 / S4.  This module moves the bytes and owns
nothing else: S1 built the plan, S2 armed the coverage, S3 built the region,
Gate 0 and the wave gate.  Here a source rank's device bytes reach a
destination rank's device bytes by exactly two routes, and no third:

* **cross-card** -- D2H into a 32 MiB staging slot of the ``/dev/shm`` region
  every rank has ``cudaHostRegister``ed, then H2D out of it on the peer.  Two
  slots per directed pair, so the producer fills slot 1 while the consumer
  drains slot 0.  Six directed pairs, 384 MiB of payload (spec section 2.2).
* **on-card** -- a raw ``cudaMalloc`` bounce buffer OUTSIDE any TMS region,
  shared by ``cudaIpcGetMemHandle`` through the region's dir area, and two D2D
  hops: the source compacts its VRAM into its own bounce, the destination
  scatters out of the imported peer pointer (spec section 1.3 step 16).  No
  PCIe key, because it crosses no link (section 3.4).

WHY A SEPARATE MODULE FROM ``weight_exchange.py``, which is what S4's file list
names.  Two reasons, and the first is the one that matters:

* **This is the only file in the exchange that calls CUDA.**  Every other half
  -- plan, coverage, region, gates -- is hermetic today and its suites run with
  ``CUDA_VISIBLE_DEVICES=""``.  Folding ctypes-on-libcudart into
  ``weight_exchange.py`` would make the plan builder's import depend on a CUDA
  runtime being loadable, and the plan builder is the one piece that has to run
  on a bare desk.
* S3 took the same deviation for the same kind of reason and stated it
  (``weight_exchange_region.py``'s module docstring); ``weight_exchange.py`` is
  already 2500 lines carrying S1+S2.

**ONE THIN ADAPTER, and it is the testability property, not a style choice.**
Every CUDA call goes through :class:`DeviceOps`.  There are two
implementations: :class:`CudartDeviceOps`, ctypes on ``libcudart``, the only
code here that can touch a GPU and the only code here with no branching; and
the test suite's fake, backed by file-mapped byte arrays and a file-mediated
IPC handle table.  That is what lets the whole transport -- batching, the slot
state machine, the semaphore handshake, the short-piece refusal, the two
on-card hops, and byte identity for every copy class including the strided 2-D
ones and the fused sub-blocks -- be proven by six real processes on a desk with
no GPU.

TWO PROBE DEFECTS FROM S0 (metal, 2026-09-09) ARE CARRIED HERE AS CODE, not as
prose:

* ``cudaIpcMemHandle_t`` is ``c_ubyte * 64``, NEVER ``c_char * 64``.  ctypes
  gives ``c_char`` arrays STRING semantics, so a read-back truncates at the
  first NUL -- and the measured handle carried only 19 non-zero bytes of 64.
  That writes a short handle and produces an open-side failure
  indistinguishable from the platform refusing IPC, i.e. a FALSE STOP-LOSS on
  this design's load-bearing premise.
* ``libcudart.so.13`` must be on the loader path.  S0 died ``exit 127`` before
  ``main()`` on exactly this; it is the same failure ``launcher.py:2906-2908``
  already names by comment for the preload hook.  :func:`find_libcudart`
  therefore searches the serving venv's cu13 directory FIRST and reports the
  path it took.  (The spec writes ``libcudart.so.12`` throughout; it was
  written before that probe.)

**THE STREAM RULE IS CHECKED, NOT ASSERTED.**  Every stream is created with
``cudaStreamCreate`` -- default, i.e. BLOCKING, flags -- never with
``cudaStreamCreateWithFlags(cudaStreamNonBlocking)``.  This is R12, pinned by
T4 at ``python/sglang/srt/weg2/tms_csrc/core.h:140-146`` (the spec cites
``core.h:84-90``; the rule moved, the rule did not change).  A non-blocking
stream does not synchronise against the legacy default stream, so a copy issued
on it can read pages a pending kernel is still writing -- silently, with no
error.  :meth:`CudartDeviceOps.create_stream` therefore reads the flags back
with ``cudaStreamGetFlags`` and refuses a non-zero value, and a source-level
test refuses the token anywhere in this file.

WHAT THIS SLICE DELIBERATELY DOES NOT DO.  It does not call ``pause`` or
``resume``, does not touch the front, does not decide a wave order, does not
create or open the region, and does not compare a pulled piece against the
destination's own restored bytes.  Those are S5 and S6; every seam they own
carries a named ``TODO`` plus a test that pins the interface so the seam cannot
drift while it is unwired.
"""

from __future__ import annotations

import ctypes
import errno
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2.weight_exchange_region import (
    Weg2XchgGateTimeout,
    Weg2XchgPlanDisagree,
    _pid_alive,
    _seal,
)

# ---------------------------------------------------------------------------
# Copy kinds.
#
# Declared here rather than imported so this module stays importable without
# torch (``weight_exchange`` imports torch for the plan builder's tensor walk).
# ``test_kind_constants_match_the_plan_builder`` pins the two sets equal: a
# rename in S1 then fails loudly here instead of making every STRIDED2D
# descriptor fall through to the FLAT branch and copy padding as payload.
# ---------------------------------------------------------------------------
FLAT = "FLAT"
STRIDED2D = "STRIDED2D"
ZEROFILL = "ZEROFILL"

#: CUDA constants, spelled once.
CUDA_MEMCPY_DEFAULT = 4
CUDA_HOST_REGISTER_PORTABLE = 0x01
CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
CUDA_IPC_HANDLE_SIZE = 64
CUDA_STREAM_DEFAULT = 0

#: The on-card bounce buffer, per rank, double-buffered like the staging slots.
#: The same 32 MiB as :data:`xr.SLOT_BYTES` and for the same measured reason
#: (E2: 32/64/128 MiB indistinguishable, <0.6 %); keeping the two equal is what
#: lets ONE batcher serve both lanes, which is why the short-piece refusal is
#: one function and not two.
ONCARD_SLOT_BYTES = xr.SLOT_BYTES
ONCARD_SLOTS = 2

#: The deepest pipeline the handshake rows are sized for.  The row area is
#: allocated from THIS number (see :data:`DIR_ONCARD_ROWS`), so a caller may
#: raise ``oncard_slots`` from the plan without moving a single offset in the
#: dir sub-layout -- the two processes would otherwise have to agree on a
#: geometry that only one of them was told.
ONCARD_SLOTS_MAX = 8

#: The largest diagonal slot :func:`plan_oncard_slot_bytes` may choose.  The
#: bounce is ``slots x slot_bytes`` of raw VRAM on the producer's card and the
#: importer maps the same allocation, so the ceiling is a residency term of
#: spec section 5, not a tuning preference: at the shipping ``ONCARD_SLOTS=2``
#: this caps the bounce at 256 MiB per card.
ONCARD_SLOT_BYTES_MAX = 128 * xr.MIB

#: The on-card lane's poll granularity.  It has no semaphore -- the diagonal is
#: absent from :data:`xr.CROSS_PAIRS` by construction -- so its handshake is a
#: bounded poll over single-writer rows, the same shape as the wave gate.
#:
#: **IT IS NO LONGER ``xr.GATE_POLL_S``, AND THAT WAS A MEASURED DEFECT.**
#: S5-pre (2026-09-09, RTX 5090, gpuq ``2xhz55``, raw in
#: ``/spinning/evidence-665-f1/xchg_s5_0909/ARMS_SUMMARY.txt``) ran 512 MiB
#: through this lane on silicon and found it **per-batch-bound, not
#: bandwidth-bound**: across an 8x change in bytes per batch (32 -> 4 MiB) the
#: consumer-side cost stayed ~0.33-0.50 ms per BATCH while the copy itself fell
#: from 0.044 ms to 0.0055 ms.  A gate fires once per WAVE; this row poll fires
#: once per BATCH, and the real diagonal is 10.28 GiB = 329 batches at a 32 MiB
#: slot -- ~118 ms against spec 6/S4's ``hop_ms <= 20``.
#:
#: THE THREE ARMS THAT SIZE THE TWO CONSTANTS BELOW, all consumer-side (the
#: producer-side instrument is 1.75x A/A noise and carries no claim):
#:
#: ===============  ====================  ==============================
#: ``ONCARD_POLL_S``  ms per batch         reading
#: ===============  ====================  ==============================
#: 500 us (the gate)  0.328-0.398          the sleep dominates
#: **50 us**          **0.182**            2.0x, far outside the 1.21x floor
#: 5 us               0.165                +0.017 for 10x the wakeups
#: ===============  ====================  ==============================
#:
#: So the sleep is 50 us -- the arm that took the 2.0x -- and the residual
#: 0.165 ms is NOT the sleep: it is the lock-step round trip itself (sync,
#: sealed-row write, the peer observing it).  That is what
#: :data:`ONCARD_SPIN_BUDGET_S` covers, and why spinning longer than it would
#: burn a core for a peer that is not coming.
ONCARD_POLL_S = 50e-6

#: The spin half of the spin-then-sleep wait, in SECONDS, from the same arms:
#: 5 us and 50 us polling measured 0.165 vs 0.182 ms per batch, i.e. the floor
#: a poll cannot go below is ~0.165 ms.  A peer that is going to answer answers
#: inside that window, so the wait spins for it and sleeps for everything
#: longer.  Spinning PAST it buys nothing measurable and costs a core.
ONCARD_SPIN_BUDGET_S = 165e-6

#: The iteration cap on that spin.  The TIME bound above is the load-bearing
#: one; this is the belt-and-braces half, because the cost of one row read is
#: not measured and a time-only bound would spin an unbounded number of times
#: on a machine where it is cheap.  Whichever bound is reached first ends the
#: spin.
ONCARD_SPIN_ITERS = 512

#: The measured per-batch cost of the lane at :data:`ONCARD_POLL_S`, in ms,
#: consumer-side -- the "poll rebind" arm of S5-pre (32 MiB slot, 16 batches,
#: 2.91 ms / 16 = 0.182).  It is the coefficient of the cost model
#: :func:`plan_oncard_slot_bytes` prices with, and it is a MEASUREMENT with a
#: named regime, never a target.
ONCARD_PER_BATCH_MS = 0.182

#: Spec section 6/S4's acceptance for the whole diagonal.
ONCARD_HOP_BUDGET_MS = 20.0

#: Modes for the on-card lane; ``host`` is the named degrade of section 3.7.
ONCARD_MODE_IPC = "ipc"
ONCARD_MODE_HOST = "host"

ENV_ONCARD_MODE = "SGLANG_WEG2_XCHG_ONCARD"

PAIR_LINE_PREFIX = "WEG2-XCHG-PAIR"
ONCARD_LINE_PREFIX = "WEG2-XCHG-ONCARD"
HOSTREG_LINE_PREFIX = "WEG2-XCHG-HOSTREG"

SHORT_PIECE_MARKER = "W54 Weg2XchgShortPiece"
ONCARD_UNAVAILABLE_MARKER = "W56 Weg2XchgOnCardUnavailable"


# ---------------------------------------------------------------------------
# The dir sub-layout.  S3 owns the region; this module owns the bytes inside
# ``[DIR_OFF, DATA_OFF)`` and says so at both ends.
# ---------------------------------------------------------------------------

#: Reserved for S6's boot-time pointer table (spec section 2.2: ~1608 x 48 B =
#: ~77 KiB).  RESERVED rather than used, so S6 lands in a hole this module has
#: already accounted for instead of one it has to be told about.
DIR_PTRTABLE_OFF = 0
DIR_PTRTABLE_BYTES = 128 * 1024

#: Six ``cudaIpcMemHandle_t``, one per rank -- the producer of a card's on-card
#: hop writes its OWN, at its OWN address.
DIR_HANDLE_OFF = DIR_PTRTABLE_OFF + DIR_PTRTABLE_BYTES
DIR_HANDLE_BYTES = xr.N_RANKS * CUDA_IPC_HANDLE_SIZE

#: The on-card handshake.  TWO row arrays, not one shared record: the producer
#: writes only its own rows and the consumer only its own, which is the law the
#: gate rows and matrix rows already obey.  A single record with two writers and
#: no atomic is a lost update, and this region has MEASURED one
#: (:meth:`xr.XchgRegion.mark_registered`, ``registered=5/6``).
#:
#: **ONE ROW PER RANK PER SLOT, not one per rank.**  MEASURED DEFECT, first
#: green remote run of S4's suite: with one row per rank the producer's batch 1
#: overwrote batch 0's ``bytes`` before the consumer had read it, and the
#: consumer -- which waits for a sequence and then compares byte counts --
#: raised W54 against a producer that had done nothing wrong
#: (``expected_bytes=4096 bytes_filled=1904``, i.e. batch 1's number answering
#: batch 0's question).  A per-slot row cannot be overwritten while the batch
#: it describes is undrained, because the producer may not reuse a slot until
#: its consumer has released it; the double buffer that makes the LANE safe is
#: what makes the SIGNAL safe, and it only works if the signal is per slot too.
DIR_ONCARD_PROD_OFF = DIR_HANDLE_OFF + DIR_HANDLE_BYTES
DIR_ONCARD_ROW_BYTES = 128
#: Sized from :data:`ONCARD_SLOTS_MAX`, not from the shipping default: the
#: pipeline depth is a per-plan knob from S5 on (the diagonal's batch count is
#: known at plan time), and a row area sized from the DEFAULT would put rank
#: r's slot 2 on top of rank r+1's slot 0 the moment a caller raised it.  The
#: dir layout is agreed by nothing at runtime -- it is two processes computing
#: the same offsets -- so its size may not depend on an argument.
DIR_ONCARD_ROWS = xr.N_RANKS * ONCARD_SLOTS_MAX
DIR_ONCARD_PROD_BYTES = DIR_ONCARD_ROWS * DIR_ONCARD_ROW_BYTES
DIR_ONCARD_CONS_OFF = DIR_ONCARD_PROD_OFF + DIR_ONCARD_PROD_BYTES
DIR_ONCARD_CONS_BYTES = DIR_ONCARD_ROWS * DIR_ONCARD_ROW_BYTES

#: THE RELEASE AREA.  One row per rank, slot 0 only, written by the on-card
#: CONSUMER once its imported peer pointer has been closed.
#:
#: MEASURED-BY-REVIEW DEFECT (S4 review + refuter, 2026-09-09, "must_fix"): the
#: producer used to return as soon as its last batch's row was written, and
#: ``run_leg``'s ``finally`` then ``cudaFree``d the bounce while the co-located
#: consumer could still be issuing D2D out of the imported pointer -- up to
#: ``ONCARD_SLOTS`` batches, 64 MiB at the shipping default, silently wrong with
#: no rc anywhere.  CUDA also forbids the EXPORTER freeing an allocation an
#: importer still has mapped.  Two handshakes close it and they are different
#: questions: the terminal DRAIN ("your last copy has landed", the consumer's
#: own rows) and this RELEASE ("your mapping is gone").  A drain alone would
#: still free under an open mapping.
DIR_ONCARD_REL_OFF = DIR_ONCARD_CONS_OFF + DIR_ONCARD_CONS_BYTES
DIR_ONCARD_REL_BYTES = DIR_ONCARD_ROWS * DIR_ONCARD_ROW_BYTES
DIR_USED_BYTES = DIR_ONCARD_REL_OFF + DIR_ONCARD_REL_BYTES
DIR_CAPACITY = xr.DATA_OFF - xr.DIR_OFF

#: ``seq`` (signed, -1 = "handle published, no batch yet"), ``bytes``,
#: ``slot_bytes``, ``wave``, ``epoch_hash``, ``pid``, ``state``.
#:
#: ``wave`` AND ``slot_bytes`` ARE BOTH MEASURED-BY-REVIEW ADDITIONS, and each
#: closes a hole the epoch hash could not:
#:
#: * ``epoch_hash`` changes per FLIP (``XchgRegion.begin_flip``), but
#:   :func:`run_leg` runs once per WAVE and every wave restarts ``seq`` at 0
#:   with a fresh bounce and a fresh handle.  Without a wave stamp, wave 2's
#:   producer saw wave 1's leftover consumer row at the same epoch, its DRAIN
#:   wait returned instantly, and it refilled a slot the consumer had not
#:   drained -- silent overwrite on every multi-wave flip.  The FILL and the
#:   handle waits had the same shape one field over.
#: * ``slot_bytes`` is the producer's batcher geometry.  It is a per-process
#:   argument that decides byte placement and was agreed by nothing; two ranks
#:   entering with different values agree on every payload below the smaller
#:   one and diverge silently above it.  Publishing it in the row makes the
#:   consumer refuse the disagreement by name (#802 rule 4).
#: S5 appends ``checksum``: the diagonal has no slot RECORD to carry it (the
#: region's ``XchgSlot`` serves the cross pairs only -- ``xr.CROSS_PAIRS`` has
#: no diagonal), so the row is the only place a co-located producer can publish
#: the sum its consumer verifies.  ``0`` means NOT COMPUTED, which is the
#: authoritative path's value and keeps that path byte-identical.
ONCARD_ROW_STRUCT = struct.Struct("<q7Q")
ONCARD_SEAL_OFF = ONCARD_ROW_STRUCT.size

ONCARD_STATE_IDLE = 0
ONCARD_STATE_ARMED = 1
ONCARD_STATE_READY = 2
ONCARD_STATE_DONE = 3
ONCARD_STATE_FAILED = 4


# ---------------------------------------------------------------------------
# Refusals.  W54 and W56 -- the two codes section 7 assigns to this slice, and
# the two still free in the branch census at 2ee844f7b8 (W51, W52, W53, W55,
# W58 and W60 are held by S1/S2/S3/S7; W57 and W59 are S6's and S5's).
# ---------------------------------------------------------------------------


class Weg2XchgShortPiece(RuntimeError):
    """W54 -- ``bytes_filled`` is not the number this consumer's plan claims.

    THE #802 RULE AT ITS PHYSICAL ROOT.  ``c26d28172106`` killed an instance
    over a corruption that had not happened, because two ranks derived their
    byte counts independently and never handshook them; #802 fixed the
    reporting and said in its own words that it did not close the physical
    case.  This is that case.  The producer writes ``bytes_filled`` AFTER its
    ``cudaStreamSynchronize`` (:meth:`xr.XchgRegion.publish`), so the number
    means "these bytes have landed"; the consumer compares it against its own
    plan-derived total BEFORE issuing a single copy.

    A short slot copied anyway is the silent-wrongness class: the destination
    takes the producer's bytes plus whatever the previous batch left in the
    slot, no error is raised anywhere, and every checksum downstream is then
    computed over the result.
    """


class Weg2XchgOnCardUnavailable(RuntimeError):
    """W56 -- ``cudaIpc`` is unusable on this build, so the on-card lane degrades.

    NOT A FAILURE OF THE FLIP when the mode was left to the arm.  Section 3.7
    degrade 3: the on-card share routes through a per-card HOST bounce instead,
    costing +0.49 s on the x4 card (+30 %) and leaving the design
    wall-neutral-to-slightly-better while keeping the ENTIRE host saving of
    section 0.2.  Decided at the launcher arm, before either group starts, so
    the mode is boot-scoped -- there is no per-flip and no per-lane fallback
    anywhere in this design (R2-1), because a lane that can change mode
    mid-boot makes ``oncard_gib`` on the flip line a number whose meaning
    changes between flips.

    IT IS RAISED, not merely logged, in exactly one case: the operator asked
    for ``ipc`` EXPLICITLY and the probe said no.  Silently handing back the
    degrade there is the "armed gate reading as a passed one" defect -- the
    boot would run a lane the operator believes is on.  Left to the default,
    the arm logs W56 and degrades, which is what section 3.7 asks for.

    S0 measured ``ipc=ok`` on this rig, so this is the armed-and-not-taken
    path.  It exists because a build, a driver or a container change can take
    it away, and a lane that degrades silently is a wall claim that has quietly
    stopped being true.
    """


def short_piece_message(
    *, pair: int, slot: int, seq: int, expected: int, filled: int,
    src: int, dst: int, epoch: str, producer_pid: int, lane: str,
    seq_filled: int,
) -> str:
    """W54's message, naming BOTH halves of the comparison it failed.

    ``seq_filled`` is not decoration.  The check is
    ``rec.seq != batch.seq or rec.bytes_filled != batch.total_bytes``, and on
    the real flip nearly every batch is a full slot -- so ``bytes_filled ==
    total_bytes`` by construction and the SEQUENCE half is the only thing that
    can catch a slot carrying a different batch.  A message that printed only
    the byte counts would say ``short_by=0`` and read as a contradiction of
    itself.  ``claim_produced`` checks the epoch and never the sequence, so
    this is the only sequence discriminator in the design.
    """
    return (
        f"{SHORT_PIECE_MARKER} lane={lane} pair={pair} slot={slot} seq={seq} "
        f"src={src} dst={dst} epoch={epoch} producer_pid={producer_pid} "
        f"expected_bytes={expected} bytes_filled={filled} "
        f"short_by={expected - filled} seq_filled={seq_filled} "
        f"seq_ok={'yes' if seq_filled == seq else 'no'} "
        f"-- the producer posted a (sequence, byte count) this consumer's plan "
        f"does not claim; NO copy was issued.  A slot copied at the producer's "
        f"own claim takes the previous batch's residue with it and raises "
        f"nothing"
    )


#: What the ``host`` degrade costs in HOST bytes, derived and not typed: one
#: bounce file per card of the three, ``ONCARD_SLOTS`` slots each.  Spec 0.2's
#: ledger term (0.38 GiB staging) does not carry it and spec 3.7 prices the
#: degrade in WALL only, so the number belongs on the line that announces the
#: degrade rather than in a comment.
ONCARD_HOST_DEGRADE_MIB = (
    (xr.N_RANKS // 2) * ONCARD_SLOTS * ONCARD_SLOT_BYTES // xr.MIB
)


def oncard_unavailable_message(*, card: str, reason: str, mode: str) -> str:
    """The W56 degrade line.

    IT MUST NAME THE TARGET IT ACTUALLY USES.  The first version of this line
    said "routes through the staging region", which is what spec 3.7 degrade 3
    and spec 6/S4 both say -- and is not what the code does.
    :data:`xr.CROSS_PAIRS` has no diagonal by construction (S3 excluded it
    because the on-card lane was never meant to take a slot), so the degrade
    cannot borrow a staging slot; it uses a per-card bounce file instead
    (:func:`oncard_host_path`).  The difference is not cosmetic: the staging
    region is already in spec 0.2's ledger and the per-card bounce is NOT, so a
    reader who trusted the old wording would conclude the degrade was
    ledger-neutral when it adds :data:`ONCARD_HOST_DEGRADE_MIB` MiB of host.
    A grep-able acceptance line that asserts spec wording about behaviour that
    does not run is worse than no line.
    """
    return (
        f"{ONCARD_UNAVAILABLE_MARKER} card={card} reason={reason} "
        f"degrade_to={mode} bounce=oncard-<card>.bin "
        f"host_add_mib={ONCARD_HOST_DEGRADE_MIB} -- the on-card share routes "
        f"through a PER-CARD host bounce file in the region directory, NOT "
        f"through the staging region (CROSS_PAIRS has no diagonal), for the "
        f"whole boot (spec 3.7 degrade 3, +0.49 s on the x4 card, +30 %); that "
        f"bounce is {ONCARD_HOST_DEGRADE_MIB} MiB of host which spec 0.2's "
        f"ledger term does not carry, and the ~34.5 GiB host saving of spec "
        f"0.2 is otherwise unaffected"
    )


def oncard_refused_message(*, card: str, reason: str) -> str:
    """W56 as a REFUSAL: ``ipc`` was asked for by name and is not available."""
    return (
        f"{ONCARD_UNAVAILABLE_MARKER} card={card} reason={reason} "
        f"degrade_to=none requested={ONCARD_MODE_IPC} -- the on-card lane was "
        f"requested as {ONCARD_MODE_IPC!r} explicitly and the probe refused "
        f"it.  Degrading here would run a boot whose operator believes the "
        f"lane is on; pass {ENV_ONCARD_MODE}={ONCARD_MODE_HOST} to take the "
        f"degrade of spec 3.7 deliberately"
    )


# ---------------------------------------------------------------------------
# The device adapter.
# ---------------------------------------------------------------------------


class DeviceOps:
    """The ONE seam every CUDA call in this module goes through.

    The method set is deliberately small and deliberately un-clever: no method
    here branches on a descriptor, a plan, a slot or a mode, because everything
    that branches must be provable on the fake.  A defect the fake cannot see
    is a defect in :class:`CudartDeviceOps` alone.

    Every method takes the CUDA device explicitly rather than reading it from
    the instance.  The current device is PER THREAD in CUDA, and this module
    runs three threads per rank on one adapter (:func:`run_leg`); an instance
    field would be a shared mutable that decides which card a copy lands on.
    """

    name = "abstract"

    # -- streams ----------------------------------------------------------
    def create_stream(self, device: int) -> int:
        raise NotImplementedError

    def stream_flags(self, stream: int) -> int:
        raise NotImplementedError

    def destroy_stream(self, stream: int) -> None:
        raise NotImplementedError

    def synchronize(self, stream: int) -> None:
        raise NotImplementedError

    # -- copies -----------------------------------------------------------
    def memcpy_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        raise NotImplementedError

    def memcpy2d_async(self, dst: int, dpitch: int, src: int, spitch: int,
                       width: int, height: int, stream: int) -> None:
        raise NotImplementedError

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        raise NotImplementedError

    # -- host pinning -----------------------------------------------------
    def host_register(self, ptr: int, nbytes: int, flags: int) -> None:
        raise NotImplementedError

    def host_unregister(self, ptr: int) -> None:
        raise NotImplementedError

    # -- the on-card lane -------------------------------------------------
    def raw_malloc(self, device: int, nbytes: int) -> int:
        """A RAW allocation, outside torch's caching allocator AND outside any
        TMS region.

        Both halves are load-bearing and both were proven on the metal by S0.
        Torch's caching allocator returns an OFFSET into a segment, which breaks
        ``cudaIpcGetMemHandle``'s base-pointer rule.  A TMS-region allocation is
        intercepted by the preload hook (``entrypoint.cpp:44-51``) and lands in
        the saver's census, which would make the exchange's own scratch a tag
        the exchange then has to transport.  S0 asserted ``tms_census_hit=0``
        on exactly this call, with the allocating thread's region flag read
        back False on both sides of it.
        """
        raise NotImplementedError

    def raw_free(self, ptr: int) -> None:
        raise NotImplementedError

    def ipc_get_handle(self, ptr: int) -> bytes:
        raise NotImplementedError

    def ipc_open_handle(self, handle: bytes) -> int:
        raise NotImplementedError

    def ipc_close_handle(self, ptr: int) -> None:
        raise NotImplementedError


def find_libcudart(venv: str = "") -> str:
    """The runtime this rig actually ships, cu13 first.

    S0 (2026-09-09) settled it by measurement: the built preload hook links
    ``libcudart.so.13`` and the serving venv is where it lives.  A cu12-only
    search -- which is what the spec's prose says -- would fail to load on the
    rig this design targets.  The system paths stay in the list LAST, so a desk
    without the venv resolves something rather than reporting "no CUDA" for
    what is a path problem.
    """
    root = venv or os.environ.get("VIRTUAL_ENV", "") or "/spinning/htsglang-gpu/.venv"
    candidates = [
        os.path.join(root, "lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"),
        os.path.join(root, "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"),
        "libcudart.so.13",
        "libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcudart.so.12",
    ]
    for path in candidates:
        if os.sep in path and not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path)
        except OSError:
            continue
        return path
    raise RuntimeError(
        "no libcudart could be loaded from " + ", ".join(candidates)
        + " -- on this rig the cu13 runtime lives in the serving venv and must "
        "be on LD_LIBRARY_PATH at LD_PRELOAD time (launcher.py:2906-2908); a "
        "process that misses it dies exit 127 before main()"
    )


class _IpcHandle(ctypes.Structure):
    # c_ubyte, NEVER c_char.  See the module docstring: a c_char array has
    # STRING semantics and truncates the handle at its first NUL.
    _fields_ = [("reserved", ctypes.c_ubyte * CUDA_IPC_HANDLE_SIZE)]


class CudartDeviceOps(DeviceOps):
    """``libcudart`` through ctypes.  No logic, so the fake can carry the proofs.

    Every entry point checks its return code and raises with the driver's own
    ``cudaGetErrorString``.  A swallowed rc here is a copy that did not happen,
    reported as one that did -- the exact shape of the no-op a checksum over a
    pre-zeroed destination exists to catch.
    """

    name = "cudart"

    def __init__(self, path: str = ""):
        self.path = path or find_libcudart()
        lib = ctypes.CDLL(self.path)
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamGetFlags.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_uint)]
        lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_size_t, ctypes.c_int,
                                        ctypes.c_void_p]
        lib.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_size_t,
                                          ctypes.c_int, ctypes.c_void_p]
        lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_size_t, ctypes.c_void_p]
        lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_uint]
        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle),
                                            ctypes.c_void_p]
        lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                             _IpcHandle, ctypes.c_uint]
        lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.lib = lib

    def _check(self, rc: int, what: str) -> None:
        if int(rc) != 0:
            msg = self.lib.cudaGetErrorString(int(rc))
            raise RuntimeError(
                f"{what} rc={rc} {msg.decode() if msg else '?'} "
                f"(libcudart={self.path})"
            )

    def set_device(self, device: int) -> None:
        self._check(self.lib.cudaSetDevice(int(device)), "cudaSetDevice")

    # -- streams ----------------------------------------------------------
    def create_stream(self, device: int) -> int:
        # cudaStreamCreate -- DEFAULT (blocking) flags.  Never
        # cudaStreamCreateWithFlags with a non-blocking flag: R12,
        # tms_csrc/core.h:140-146.  The flags are read BACK below rather than
        # trusted, because "we called the blocking constructor" is exactly the
        # kind of claim that survives a refactor while ceasing to be true.
        self.set_device(device)
        handle = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(handle)),
                    "cudaStreamCreate")
        stream = int(handle.value or 0)
        flags = self.stream_flags(stream)
        if flags != CUDA_STREAM_DEFAULT:
            self.destroy_stream(stream)
            raise RuntimeError(
                f"cudaStreamCreate returned a stream with flags={flags:#x}, "
                f"not cudaStreamDefault ({CUDA_STREAM_DEFAULT}).  A "
                f"non-blocking stream does not synchronise against the legacy "
                f"default stream, so a copy issued on it can read pages a "
                f"pending kernel is still writing -- silently, with no error "
                f"(R12, tms_csrc/core.h:140-146)"
            )
        return stream

    def stream_flags(self, stream: int) -> int:
        out = ctypes.c_uint(0)
        self._check(
            self.lib.cudaStreamGetFlags(ctypes.c_void_p(stream), ctypes.byref(out)),
            "cudaStreamGetFlags",
        )
        return int(out.value)

    def destroy_stream(self, stream: int) -> None:
        self._check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)),
                    "cudaStreamDestroy")

    def synchronize(self, stream: int) -> None:
        self._check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)),
                    "cudaStreamSynchronize")

    # -- copies -----------------------------------------------------------
    def memcpy_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                     nbytes, CUDA_MEMCPY_DEFAULT,
                                     ctypes.c_void_p(stream)),
            "cudaMemcpyAsync",
        )

    def memcpy2d_async(self, dst: int, dpitch: int, src: int, spitch: int,
                       width: int, height: int, stream: int) -> None:
        self._check(
            self.lib.cudaMemcpy2DAsync(ctypes.c_void_p(dst), dpitch,
                                       ctypes.c_void_p(src), spitch,
                                       width, height, CUDA_MEMCPY_DEFAULT,
                                       ctypes.c_void_p(stream)),
            "cudaMemcpy2DAsync",
        )

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        self._check(
            self.lib.cudaMemsetAsync(ctypes.c_void_p(dst), int(value), nbytes,
                                     ctypes.c_void_p(stream)),
            "cudaMemsetAsync",
        )

    # -- host pinning -----------------------------------------------------
    def host_register(self, ptr: int, nbytes: int, flags: int) -> None:
        self._check(self.lib.cudaHostRegister(ctypes.c_void_p(ptr), nbytes, flags),
                    "cudaHostRegister")

    def host_unregister(self, ptr: int) -> None:
        self._check(self.lib.cudaHostUnregister(ctypes.c_void_p(ptr)),
                    "cudaHostUnregister")

    # -- the on-card lane -------------------------------------------------
    def raw_malloc(self, device: int, nbytes: int) -> int:
        self.set_device(device)
        out = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(out), nbytes), "cudaMalloc")
        return int(out.value or 0)

    def raw_free(self, ptr: int) -> None:
        self._check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def ipc_get_handle(self, ptr: int) -> bytes:
        handle = _IpcHandle()
        self._check(
            self.lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr)),
            "cudaIpcGetMemHandle",
        )
        # string_at over the c_ubyte array: all 64 bytes, NULs included.
        return ctypes.string_at(ctypes.byref(handle), CUDA_IPC_HANDLE_SIZE)

    def ipc_open_handle(self, handle: bytes) -> int:
        if len(handle) != CUDA_IPC_HANDLE_SIZE:
            raise ValueError(
                f"an IPC handle is exactly {CUDA_IPC_HANDLE_SIZE} bytes, got "
                f"{len(handle)} -- a short handle is the c_char truncation "
                f"defect S0 caught at the desk, not a platform refusal"
            )
        blob = _IpcHandle()
        ctypes.memmove(ctypes.byref(blob), handle, CUDA_IPC_HANDLE_SIZE)
        out = ctypes.c_void_p()
        self._check(
            self.lib.cudaIpcOpenMemHandle(
                ctypes.byref(out), blob, CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
            "cudaIpcOpenMemHandle",
        )
        return int(out.value or 0)

    def ipc_close_handle(self, ptr: int) -> None:
        self._check(self.lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr)),
                    "cudaIpcCloseMemHandle")


# ---------------------------------------------------------------------------
# Semaphores.  S3 creates and unlinks the 24 names; this slice opens, waits and
# posts them, exactly as ``create_semaphores``' docstring says.
# ---------------------------------------------------------------------------


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class SemSet:
    """The 24 named semaphores of one boot, opened once per process.

    NEVER ``O_CREAT`` here.  The launcher created them with ``O_EXCL`` after
    unlinking, so a creating ``sem_open`` in a rank would silently adopt a name
    the launcher had not made -- and POSIX ignores the initial value for an
    existing name, so the adoption would be invisible and the first producer
    would block until the 120 s fence budget.  Opening without ``O_CREAT``
    turns "the launcher did not run" into an ``ENOENT`` right here.
    """

    def __init__(self, boot_nonce: str):
        self.boot_nonce = str(boot_nonce)
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        lib.sem_open.restype = ctypes.c_void_p
        lib.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.sem_trywait.argtypes = [ctypes.c_void_p]
        lib.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Timespec)]
        lib.sem_post.argtypes = [ctypes.c_void_p]
        lib.sem_close.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._handles: Dict[Tuple[int, int, str], int] = {}
        self._lock = threading.Lock()

    def handle(self, pair: int, slot: int, kind: str) -> int:
        key = (int(pair), int(slot), kind)
        got = self._handles.get(key)
        if got is not None:
            return got
        with self._lock:
            got = self._handles.get(key)
            if got is not None:
                return got
            name = xr.sem_name(self.boot_nonce, pair, slot, kind)
            ctypes.set_errno(0)
            raw = self._lib.sem_open(name.encode("ascii"), 0)
            if raw in (None, 0, ctypes.c_void_p(-1).value):
                err = ctypes.get_errno()
                raise OSError(
                    err,
                    f"sem_open({name}) without O_CREAT failed: "
                    f"{os.strerror(err)} -- the launcher creates all 24 before "
                    f"either group starts "
                    f"(weight_exchange_region.create_semaphores); creating one "
                    f"here would adopt a name with an unknown count",
                )
            self._handles[key] = int(raw)
            return int(raw)

    # -- the two raw calls, isolated so a test can inject an errno -----------
    #
    # ``use_errno=True`` on the CDLL is what makes ``ctypes.get_errno()`` the
    # errno of THIS call rather than of whatever the interpreter did next.  It
    # was set from the start and then never read, which is the whole defect
    # below.
    def _trywait_once(self, handle) -> int:
        ctypes.set_errno(0)
        if self._lib.sem_trywait(handle) == 0:
            return 0
        return ctypes.get_errno()

    def _timedwait_once(self, handle, ts) -> int:
        ctypes.set_errno(0)
        if self._lib.sem_timedwait(handle, ctypes.byref(ts)) == 0:
            return 0
        return ctypes.get_errno()

    def trywait(self, pair: int, slot: int, kind: str) -> bool:
        handle = ctypes.c_void_p(self.handle(pair, slot, kind))
        while True:
            err = self._trywait_once(handle)
            if err == 0:
                return True
            if err == errno.EINTR:
                continue
            if err in (errno.EAGAIN, 0):
                return False
            raise OSError(err, f"sem_trywait(pair={pair} slot={slot} "
                               f"kind={kind}) failed: {os.strerror(err)}")

    def timedwait(self, pair: int, slot: int, kind: str, budget_s: float) -> bool:
        """Wait for one semaphore, or report a REAL timeout.

        **EINTR IS NOT A TIMEOUT, and reading it as one is group-fatal.**
        MEASURED-BY-REVIEW DEFECT (S4 refuter, 2026-09-09): this returned
        ``rc == 0`` and never looked at errno, so any signal delivered to a
        producer or consumer thread during a slot wait -- and these run inside
        a live ``launch_server`` beside torch's watchdogs and child reaping --
        came back as ``False``, which :func:`_take_slot` turns into
        ``W53 Weg2XchgGateTimeout ... no producer posted this slot within the
        fence budget``, naming a peer that is perfectly healthy.
        :func:`run_leg` then votes ``ok=False``, which is group-fatal: W29 ->
        front W4 -> ``do_stop``.  A guard that cries wolf on the healthy path
        is the worst kind there is, and this file names that rule in
        ``test_batches_are_the_same_whether_the_side_coalesced_or_not`` -- one
        wait over from where it was not applied.

        The retry is against the SAME ABSOLUTE DEADLINE (``ts`` is computed
        once, outside the loop, and ``sem_timedwait`` takes an absolute
        ``CLOCK_REALTIME``), so a storm of signals cannot extend the fence
        budget by even one poll.  The neighbouring :func:`_await_oncard` gets
        this right by construction because it re-polls against its own
        deadline; this is the same property, spelled.
        """
        handle = ctypes.c_void_p(self.handle(pair, slot, kind))
        deadline = time.clock_gettime(time.CLOCK_REALTIME) + float(budget_s)
        ts = _Timespec(int(deadline), int((deadline % 1.0) * 1e9))
        while True:
            err = self._timedwait_once(handle, ts)
            if err == 0:
                return True
            if err == errno.EINTR:
                continue
            if err in (errno.ETIMEDOUT, 0):
                return False
            raise OSError(err, f"sem_timedwait(pair={pair} slot={slot} "
                               f"kind={kind}) failed: {os.strerror(err)}")

    def post(self, pair: int, slot: int, kind: str) -> None:
        if self._lib.sem_post(ctypes.c_void_p(self.handle(pair, slot, kind))) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"sem_post failed: {os.strerror(err)}")

    def close(self) -> None:
        with self._lock:
            for raw in self._handles.values():
                self._lib.sem_close(ctypes.c_void_p(raw))
            self._handles.clear()


# ---------------------------------------------------------------------------
# Batching: the plan turned into slot-sized units, IDENTICALLY on both sides.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """One copy inside one slot.

    ``slot_off`` is where it sits in the staging slot; ``src_off``/``dst_off``
    are byte offsets from the descriptor's own pointers.  A STRIDED2D piece
    keeps its pitches: on the way IN it compacts (``dpitch = run_bytes``), on
    the way OUT it scatters (``spitch = run_bytes``), so only payload bytes ever
    cross the link.  That is the form measured at 1.000x flat (spec 1.4.3).
    """

    desc_index: int
    kind: str
    nbytes: int
    slot_off: int
    src_off: int
    dst_off: int
    rows: int = 1
    run_bytes: int = 0
    spitch: int = 0
    dpitch: int = 0


@dataclass(frozen=True)
class SlotBatch:
    """One slot's worth of pieces, in issue order."""

    seq: int
    pieces: Tuple[Piece, ...]
    total_bytes: int

    @property
    def slot(self) -> int:
        """Which of the pair's two slots this batch uses.

        ``seq % SLOTS_PER_PAIR`` and nothing else: the alternation IS the
        double buffering, and deriving it here rather than in both loops is
        what lets producer and consumer agree without a message.
        """
        return self.seq % xr.SLOTS_PER_PAIR

    def checksum(self, read_bytes: Optional[Callable[[int, int], int]] = None,
                 base: int = 0) -> int:
        """The ``uint8_checksum`` of this batch's STAGED bytes, or 0.

        WIRED IN S5 (this was ``TODO(S5)`` and returned 0 unconditionally).
        ``read_bytes(addr, nbytes) -> int`` is the caller's summer -- on the
        metal a device-side sum over a scratch tensor, in the double a byte
        sum over the fake's storage.  ``None`` keeps the authoritative path
        BYTE-IDENTICAL to today: no read, no sync, no cost, and ``0`` keeps its
        meaning of "not computed".

        THE RANGE IS ``[base, base + total_bytes)`` AND THAT IS A PROPERTY OF
        THE BATCHER, not an assumption: :func:`batch_descs` packs pieces at
        ``slot_off = cur_bytes``, a running sum, so a batch occupies the front
        of its slot contiguously and a 2-D piece is COMPACTED into it
        (``dpitch = run_bytes``).  Summing the range is therefore summing
        exactly the payload -- never slot padding, which is stale bytes from an
        older batch and would make an honest transport report a mismatch.
        ``test_the_checksum_covers_the_payload_and_not_the_slot_padding`` is
        what holds that.

        IT DOES NOT CLASSIFY AND IT DOES NOT RAISE.  Section 3.3 rule 3 says
        ``checksum_is_representable`` (``model_executor/weights_arena.py:133``)
        must be asked before any mismatch is REPORTED, and that question needs
        a context this module does not have (it must stay importable without
        torch).  The transport hands the two numbers to its
        ``on_checksum`` callback; the shadow decides what they mean.
        """
        if read_bytes is None:
            return 0
        return int(read_bytes(int(base), int(self.total_bytes)))


#: What ``SlotBatch.checksum`` returns when no summer was supplied.  A named
#: constant because ``0`` is also a legal checksum (an all-zero payload), and
#: the two are told apart by whether a summer was passed -- never by the value.
CHECKSUM_NOT_COMPUTED = 0


@dataclass(frozen=True)
class ChecksumReport:
    """One staged batch's producer checksum beside the consumer's own.

    HANDED TO A CALLBACK, NEVER RAISED, and that is the shadow's zero-authority
    rule expressed in the transport's own signature: this module cannot tell a
    corruption from a framing error (that is what ``checksum_is_representable``
    is for) and a module that cannot classify a fault must not act on one.

    ``expected`` is the PRODUCER's published number and ``got`` is this
    consumer's own sum over the same staged range.  ``match`` is the plain
    comparison; everything else about what it MEANS belongs to the caller.
    """

    lane: str
    pair: int
    slot: int
    seq: int
    nbytes: int
    expected: int
    got: int

    @property
    def match(self) -> bool:
        return int(self.expected) == int(self.got)


def batch_descs(
    descs: Sequence[object],
    slot_bytes: int = xr.SLOT_BYTES,
    *,
    first_seq: int = 0,
) -> List[SlotBatch]:
    """Turn one directed pair's descriptors into slot-sized batches.

    **DETERMINISTIC FROM THE PLAN ALONE, and that is the whole contract.**
    Producer and consumer never exchange a batch description: each calls this
    on its own copy of the same descriptor list -- the list whose ``plan_id``
    Gate 0 already agreed across all six ranks -- and gets the same batches, in
    the same order, with the same ``total_bytes``.  W54 is what happens when
    that stops being true, which is exactly why the comparison is against the
    consumer's OWN derivation and not against a number the producer also chose.

    THE PACKING IS OVER A BYTE STREAM, NOT OVER DESCRIPTORS, and that is what
    makes it survive S1's coalescer being ASYMMETRIC.  ``coalesce`` merges only
    where both address spaces are adjacent, and a rank holds pointers for one
    side only, so the source and the destination can legitimately merge
    different pieces (``build_plan`` says so where it computes ``plan_id`` from
    the RAW list).  Because this function packs greedily by RUNNING BYTE COUNT
    over descriptors that are a partition of the same ordered byte stream, a
    merged pair and its two halves land at the same slot offsets and produce
    the same batch boundaries.  ``test_batches_are_the_same_whether_the_side_
    coalesced_or_not`` proves it rather than leaving it argued.

    A FLAT descriptor larger than a slot splits at slot boundaries; a STRIDED2D
    descriptor splits by ROWS, a row being the smallest unit whose pitch
    arithmetic stays exact.  A ZEROFILL descriptor never enters a batch: it has
    no source, so there is nothing to stage -- the destination memsets it
    locally (:func:`apply_zerofill`).
    """
    if slot_bytes <= 0:
        raise ValueError(f"slot_bytes must be positive, not {slot_bytes!r}")
    batches: List[SlotBatch] = []
    cur: List[Piece] = []
    cur_bytes = 0
    seq = int(first_seq)

    def flush() -> None:
        nonlocal cur, cur_bytes, seq
        if not cur:
            return
        batches.append(SlotBatch(seq, tuple(cur), cur_bytes))
        seq += 1
        cur = []
        cur_bytes = 0

    for index, desc in enumerate(descs):
        kind = desc.kind
        if kind == ZEROFILL:
            continue
        if kind == FLAT:
            remaining = int(desc.nbytes)
            done = 0
            while remaining > 0:
                room = slot_bytes - cur_bytes
                if room <= 0:
                    flush()
                    room = slot_bytes
                take = min(room, remaining)
                cur.append(Piece(
                    desc_index=index, kind=FLAT, nbytes=take, slot_off=cur_bytes,
                    src_off=int(desc.src_off) + done,
                    dst_off=int(desc.dst_off) + done,
                    rows=1, run_bytes=take,
                ))
                cur_bytes += take
                done += take
                remaining -= take
            continue
        if kind != STRIDED2D:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree batch: descriptor {index} "
                f"({getattr(desc, 'param_name', '?')}) has kind {kind!r}, which "
                f"is none of {FLAT}/{STRIDED2D}/{ZEROFILL} -- the transport "
                f"refuses a shape it cannot price rather than falling through "
                f"to the flat branch and copying padding as payload"
            )
        run = int(desc.run_bytes)
        if run <= 0 or run > slot_bytes:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree batch: descriptor {index} "
                f"({getattr(desc, 'param_name', '?')}) has run_bytes={run}, "
                f"which does not fit a {slot_bytes}-byte slot; a run is the "
                f"smallest indivisible unit of a 2-D copy, so this plan cannot "
                f"be staged at this slot size"
            )
        rows_left = int(desc.rows)
        row0 = 0
        while rows_left > 0:
            room = slot_bytes - cur_bytes
            if room < run:
                flush()
                room = slot_bytes
            take = min(rows_left, room // run)
            cur.append(Piece(
                desc_index=index, kind=STRIDED2D, nbytes=take * run,
                slot_off=cur_bytes,
                src_off=int(desc.src_off) + row0 * int(desc.spitch),
                dst_off=int(desc.dst_off) + row0 * int(desc.dpitch),
                rows=take, run_bytes=run,
                spitch=int(desc.spitch), dpitch=int(desc.dpitch),
            ))
            cur_bytes += take * run
            row0 += take
            rows_left -= take
    flush()
    return batches


def pair_descs(descs: Iterable[object], src_rank: int, dst_rank: int) -> List[object]:
    """One directed pair's descriptors, IN PLAN ORDER.

    Plan order, never re-sorted here.  ``build_plan`` sorts once, by
    ``(wave, dst_rank, param_name, dst_off, src_rank)``, and both processes
    inherit that order; a second sort on a different key in one of them would
    move every batch boundary and make every slot short.

    ``src_rank``/``dst_rank`` are GROUP-LOCAL rank indices, which is also the
    positional card index: both groups get the same ``CUDA_VISIBLE_DEVICES``
    uuid string, so rank ``n`` of either group runs on ``cards[n]``
    (``weight_exchange.GroupLayout``).  That is why they index
    :data:`xr.CROSS_PAIRS` directly, and why ``src_rank == dst_rank`` is the
    on-card lane.
    """
    return [d for d in descs
            if int(d.src_rank) == int(src_rank) and int(d.dst_rank) == int(dst_rank)]


# ---------------------------------------------------------------------------
# Host staging registration.
# ---------------------------------------------------------------------------


@dataclass
class HostRegistration:
    registered: bool
    reason: str
    address: int
    nbytes: int
    count: int

    def line(self, rank_row: int) -> str:
        return (
            f"{HOSTREG_LINE_PREFIX} row={rank_row} "
            f"registered={'yes' if self.registered else 'no'} "
            f"addr={self.address:#x} bytes={self.nbytes} "
            f"count={self.count}/{xr.N_RANKS} reason={self.reason or 'ok'}"
        )


def register_region(
    region: xr.XchgRegion,
    ops: DeviceOps,
    row: int,
    *,
    log: Callable[[str], None],
) -> HostRegistration:
    """``cudaHostRegister`` the whole region on this rank, once.

    ADDENDUM 2 arm R measured a registered ``/dev/shm`` MAP_SHARED mapping
    byte-for-byte equal to ``cudaHostAlloc`` (14.299/14.450 vs 14.296/14.449
    GB/s on the 5090); NOT registering costs 7.5 % / 3.3 %.  So a failure here
    is a performance regression, not a correctness one -- and that is precisely
    why it must not be swallowed.  This returns a result and logs a NAMED line
    instead of raising, and it does NOT call ``mark_registered`` on failure, so
    the region line's ``registered=<n>/6`` denominator keeps telling the truth
    rather than reporting an armed region that is pinned on five ranks.

    No new W-code: the denominator IS the instrument here, and inventing a
    refusal for a 7.5 % regression would be a gate that has to be disarmed the
    first time a container drops ``CAP_IPC_LOCK``.

    ``cudaHostRegisterPortable`` per spec section 6/S4.  ADDENDUM 2's probe used
    flags 0; portable is the stricter of the two -- the mapping is pinned for
    every context in the process, which is what a rank running several streams
    on one device needs -- and it is what the spec names.
    """
    address = region.base_address()
    nbytes = xr.REGION_BYTES
    try:
        ops.host_register(address, nbytes, CUDA_HOST_REGISTER_PORTABLE)
    except Exception as exc:  # noqa: BLE001 -- the reason IS the payload
        result = HostRegistration(False, f"{type(exc).__name__}: {exc}",
                                  address, nbytes, region.registered_count())
        log(result.line(row))
        return result
    count = region.mark_registered(row, log=log)
    result = HostRegistration(True, "", address, nbytes, count)
    log(result.line(row))
    return result


def unregister_region(ops: DeviceOps, reg: HostRegistration) -> None:
    if reg.registered:
        ops.host_unregister(reg.address)


# ---------------------------------------------------------------------------
# The cross-card lane.
# ---------------------------------------------------------------------------


@dataclass
class PairStats:
    src_card: int
    dst_card: int
    src_uuid: str
    dst_uuid: str
    bytes_moved: int = 0
    strided_bytes: int = 0
    pieces: int = 0
    batches: int = 0
    slot_waits: int = 0
    slot_wait_s: float = 0.0
    elapsed_s: float = 0.0

    def line(self) -> str:
        """The acceptance line of spec section 6/S4, verbatim in its tokens.

        ``gbs`` is decimal GB/s over the bytes COPIED -- the convention of
        ``xfer_probe_0908.gbs()`` and of S0's 762.591 -- so this number is
        comparable to E1-E5 and to the S0 record and to nothing else.  A
        zero-length leg prints ``gbs=0.000`` rather than dividing by zero, and
        its denominator is visible right there in ``bytes_mib=0.00``.

        R10: this line is keyed by SOURCE and DESTINATION uuid, not aggregated.
        ``3080-x8 -> 3080-x4`` took 1.5730 s of a 1.5752 s wall, so a
        regression confined to one pair is a full flip-time regression that is
        invisible in any average.

        **``gbs`` IS CONSUMER-PACED, and the acceptance must say so.**
        ``elapsed_s`` spans the whole leg including the ``empty``/``full`` slot
        waits, so this is a leg rate, not a link rate: a producer held by a
        slow peer prints a low ``gbs`` with nothing wrong on its own side.
        That is deliberate -- the leg rate is what the flip budget is spent in
        -- but spec 6/S4's threshold ("within 10 % of 6.375-6.538 GB/s") is a
        LINK figure, so the comparison must be made against
        ``bytes_moved / (elapsed_s - slot_wait_s)``.  Both terms are on the
        line for exactly that reason; the subtraction is the reader's, and
        naming it here is what keeps it from being forgotten at S5.
        """
        ms = self.elapsed_s * 1e3
        gbs = (self.bytes_moved / self.elapsed_s / 1e9) if self.elapsed_s > 0 else 0.0
        return (
            f"{PAIR_LINE_PREFIX} src={self.src_uuid} dst={self.dst_uuid} "
            f"bytes_mib={self.bytes_moved / xr.MIB:.2f} pieces={self.pieces} "
            f"strided_mib={self.strided_bytes / xr.MIB:.2f} ms={ms:.3f} "
            f"gbs={gbs:.3f} slot_waits={self.slot_waits} "
            f"slot_wait_ms={self.slot_wait_s * 1e3:.3f}"
        )


def require_slot_bytes(ceiling: int, slot_bytes: int, *, what: str) -> int:
    """Refuse a batcher slot size that does not fit the storage it will be
    written into -- BEFORE the first copy, not after it.

    MEASURED-BY-REVIEW DEFECT (S4 review + refuter, 2026-09-09, "must_fix"):
    ``slot_bytes`` was a caller-supplied knob bounded only by ``> 0``, while
    the staging slots are exactly ``xr.SLOT_BYTES`` apart
    (``XchgRegion.data_offset``).  A ``slot_bytes`` above that packs batches of
    that size, writes them at ``slot_address(pair, slot) + slot_off``, syncs --
    and only THEN does ``XchgRegion.publish`` refuse ``bytes_filled >
    SLOT_BYTES``.  By that point the overflow is already in the NEXT slot's
    payload, which belongs to another directed pair and is live.  The one test
    that named this hazard proved only that the batcher respects the number it
    was handed, which is the half that was never in doubt.

    ``XchgRegion.open`` validates the region HEADER's slot size against the
    module constant; that validation was simply never connected to this
    argument.  Here it is, at both cross-lane entry points and at
    :func:`run_leg`, for both of its knobs.
    """
    value = int(slot_bytes)
    if value <= 0 or value > int(ceiling):
        raise Weg2XchgPlanDisagree(
            f"W52 Weg2XchgPlanDisagree {what}: slot_bytes={value} is not in "
            f"1..{int(ceiling)}.  A batch larger than the slot it is written "
            f"into is issued past the end of that slot, into the next pair's "
            f"live payload, and `publish` only notices afterwards -- the "
            f"bytes have landed by then"
        )
    return value


def require_oncard_slots(slots: int, *, what: str) -> int:
    """Refuse a diagonal pipeline depth the handshake rows cannot address.

    The sibling of :func:`require_slot_bytes`, and the same defect one field
    over: ``ONCARD_SLOTS`` became a per-plan knob in S5 (the diagonal's batch
    count is known at plan time, so a deeper pipeline is a real choice), and
    the row area is sized ONCE from :data:`ONCARD_SLOTS_MAX` because two
    processes compute its offsets independently.  A depth above the max is not
    a slow lane -- it is rank ``r``'s slot ``k`` landing on rank ``r+1``'s slot
    0, with both sides agreeing on the wrong address.
    """
    value = int(slots)
    if value < 1 or value > ONCARD_SLOTS_MAX:
        raise Weg2XchgPlanDisagree(
            f"W52 Weg2XchgPlanDisagree {what}: oncard_slots={value} is not in "
            f"1..{ONCARD_SLOTS_MAX}.  The on-card handshake row area is sized "
            f"once, from ONCARD_SLOTS_MAX, so a deeper pipeline addresses "
            f"another rank's rows -- identically on both sides, which is why "
            f"nothing would disagree about it"
        )
    return value


def _take_slot(sems: SemSet, pair: int, slot: int, kind: str, budget_s: float,
               stats: PairStats, on_timeout: Callable[[], BaseException]) -> None:
    """Take one semaphore, counting whether it actually blocked.

    ``sem_trywait`` FIRST: that is what makes ``slot_waits`` exact rather than a
    threshold over an elapsed time.  A wait that did not block is not a wait,
    and counting it as one would leave ``slot_wait_ms`` as unfalsifiable as
    ``lock_wait_ms`` already is (R2-10) -- a number that cannot be non-zero is
    not a measurement.
    """
    if sems.trywait(pair, slot, kind):
        return
    started = time.perf_counter()
    ok = sems.timedwait(pair, slot, kind, budget_s)
    stats.slot_waits += 1
    stats.slot_wait_s += time.perf_counter() - started
    if not ok:
        raise on_timeout()


def run_producer_pair(
    region: xr.XchgRegion,
    sems: SemSet,
    ops: DeviceOps,
    stream: int,
    *,
    pair: int,
    descs: Sequence[object],
    stats: PairStats,
    budget_s: Optional[float] = None,
    slot_bytes: int = xr.SLOT_BYTES,
    checksum_bytes: Optional[Callable[[int, int], int]] = None,
) -> PairStats:
    """D2H one directed pair's descriptors into the staging slots.

    THE ORDER IS THE LAW, and it is what makes ``bytes_filled`` mean anything:
    fill, then **one** ``cudaStreamSynchronize`` for the whole slot, then
    ``publish`` (which writes ``bytes_filled``), then ``sem_post(full)``.  One
    sync per SLOT, never per copy -- E4 measured a per-copy sync at 2.04x, and
    it would also turn ``issue_ms`` into the whole leg.

    A refusal raised here leaves the slot FILLING with ``full`` UNPOSTED, which
    :meth:`xr.XchgRegion.publish` documents: the caller must vote ``ok=False``
    into its gate row rather than swallow it, or the consumer waits the full
    fence budget for bytes that will never come.  :func:`run_leg` is what does
    that, which is why its ``vote_failure`` argument is required.
    """
    budget = xr.fence_budget_s() if budget_s is None else float(budget_s)
    slot_bytes = require_slot_bytes(region.header()["slot_bytes"], slot_bytes,
                                    what=f"produce pair={pair}")
    batches = batch_descs(descs, slot_bytes)
    src, dst = xr.CROSS_PAIRS[pair]
    started = time.perf_counter()
    for batch in batches:
        slot = batch.slot

        def expired(slot=slot, batch=batch) -> BaseException:
            return Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout slot pair={pair} slot={slot} "
                f"kind=empty epoch={region.epoch} src_card={src} dst_card={dst} "
                f"seq={batch.seq} budget_s={budget} -- the consumer has not "
                f"drained this slot; the producer is blocked holding bytes it "
                f"cannot place"
            )

        _take_slot(sems, pair, slot, "empty", budget, stats, expired)
        region.begin_fill(pair, slot, batch.seq)
        base = region.slot_address(pair, slot)
        for piece in batch.pieces:
            desc = descs[piece.desc_index]
            src_ptr = int(desc.src_ptr) + piece.src_off
            if piece.kind == FLAT:
                ops.memcpy_async(base + piece.slot_off, src_ptr, piece.nbytes,
                                 stream)
            else:
                # COMPACTING D2H: dpitch is the run, so the slot holds payload
                # only and the link carries no padding.
                ops.memcpy2d_async(base + piece.slot_off, piece.run_bytes,
                                   src_ptr, piece.spitch,
                                   piece.run_bytes, piece.rows, stream)
                stats.strided_bytes += piece.nbytes
            stats.pieces += 1
        ops.synchronize(stream)
        # AFTER the sync, like `bytes_filled` and for the identical reason: a
        # checksum taken over issued-but-not-landed bytes is a checksum of the
        # PREVIOUS batch, and it would be published as this one's.
        region.publish(pair, slot, batch.total_bytes,
                       checksum=batch.checksum(checksum_bytes, base))
        sems.post(pair, slot, "full")
        stats.bytes_moved += batch.total_bytes
        stats.batches += 1
    stats.elapsed_s = time.perf_counter() - started
    return stats


def run_consumer_pair(
    region: xr.XchgRegion,
    sems: SemSet,
    ops: DeviceOps,
    stream: int,
    *,
    pair: int,
    descs: Sequence[object],
    stats: PairStats,
    budget_s: Optional[float] = None,
    slot_bytes: int = xr.SLOT_BYTES,
    checksum_bytes: Optional[Callable[[int, int], int]] = None,
    on_checksum: Optional[Callable[[ChecksumReport], None]] = None,
) -> PairStats:
    """H2D one directed pair's descriptors out of the staging slots.

    THE SHORT-PIECE CHECK IS BEFORE THE FIRST COPY -- not after it, and not
    beside it.  ``rec.bytes_filled`` is the producer's post-sync claim;
    ``batch.total_bytes`` is this rank's own derivation from the plan Gate 0
    agreed.  If they differ the transport raises W54 and issues NOTHING.
    """
    budget = xr.fence_budget_s() if budget_s is None else float(budget_s)
    slot_bytes = require_slot_bytes(region.header()["slot_bytes"], slot_bytes,
                                    what=f"consume pair={pair}")
    batches = batch_descs(descs, slot_bytes)
    src, dst = xr.CROSS_PAIRS[pair]
    started = time.perf_counter()
    for batch in batches:
        slot = batch.slot

        def expired(slot=slot, batch=batch) -> BaseException:
            return Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout slot pair={pair} slot={slot} "
                f"kind=full epoch={region.epoch} src_card={src} dst_card={dst} "
                f"seq={batch.seq} budget_s={budget} -- no producer posted this "
                f"slot within the fence budget"
            )

        _take_slot(sems, pair, slot, "full", budget, stats, expired)
        rec = region.claim_produced(pair, slot)
        if rec is None:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree consume pair={pair} slot={slot} "
                f"seq={batch.seq} epoch={region.epoch}: the slot's `full` "
                f"semaphore was posted but the record is not PRODUCED at this "
                f"flip's epoch -- a post and a publish that do not agree"
            )
        if rec.seq != batch.seq or rec.bytes_filled != batch.total_bytes:
            raise Weg2XchgShortPiece(short_piece_message(
                lane="cross", pair=pair, slot=slot, seq=batch.seq,
                expected=batch.total_bytes, filled=rec.bytes_filled,
                seq_filled=rec.seq,
                src=src, dst=dst, epoch=region.epoch,
                producer_pid=rec.producer_pid,
            ))
        base = region.slot_address(pair, slot)
        _report_checksum(on_checksum, checksum_bytes, batch, base,
                         lane="cross", pair=pair, slot=slot,
                         published=rec.checksum)
        for piece in batch.pieces:
            desc = descs[piece.desc_index]
            dst_ptr = int(desc.dst_ptr) + piece.dst_off
            if piece.kind == FLAT:
                ops.memcpy_async(dst_ptr, base + piece.slot_off, piece.nbytes,
                                 stream)
            else:
                # SCATTERING H2D: spitch is the run (the slot is compact),
                # dpitch is the destination arena's pitch.
                ops.memcpy2d_async(dst_ptr, piece.dpitch,
                                   base + piece.slot_off, piece.run_bytes,
                                   piece.run_bytes, piece.rows, stream)
                stats.strided_bytes += piece.nbytes
            stats.pieces += 1
        ops.synchronize(stream)
        region.release_slot(pair, slot)
        sems.post(pair, slot, "empty")
        stats.bytes_moved += batch.total_bytes
        stats.batches += 1
    stats.elapsed_s = time.perf_counter() - started
    return stats


def _report_checksum(on_checksum, checksum_bytes, batch, base: int, *,
                     lane: str, pair: int, slot: int, published: int) -> None:
    """Compare the staged bytes against the producer's published sum, and TELL.

    Silent when either half is absent -- no summer, or a producer that
    published ``CHECKSUM_NOT_COMPUTED`` -- because a comparison against a
    number nobody computed is the "unarmed gate reading as a passed one" shape
    the spec forbids in section 4.2.  It never raises: on the authoritative
    path there is no summer at all, and in the shadow a mismatch is COUNTED,
    which is what zero authority means when the finding is real.
    """
    if on_checksum is None or checksum_bytes is None:
        return
    if int(published) == CHECKSUM_NOT_COMPUTED:
        return
    got = batch.checksum(checksum_bytes, base)
    on_checksum(ChecksumReport(lane=lane, pair=int(pair), slot=int(slot),
                               seq=int(batch.seq), nbytes=int(batch.total_bytes),
                               expected=int(published), got=int(got)))


def apply_zerofill(ops: DeviceOps, stream: int, descs: Sequence[object],
                   dst_rank: int) -> int:
    """``cudaMemsetAsync`` every ZEROFILL range this rank owns.

    The vocab pad rows (spec section 2.2): D pads to ``64 * tp = 192`` while P
    at tp=1 pads to 64 and 248320 is already a multiple of 64, so D rank 2 owns
    all 128 pad rows and they have NO VRAM source and NO checkpoint source.
    They are not a transport case, they are an initialisation case, and giving
    them a descriptor rather than leaving them undefined is what makes W58's
    "every destination byte has a source or a ZEROFILL" checkable at all.

    ``dst_rank`` is the GROUP-LOCAL rank, matching ``XchgDesc.dst_rank`` -- not
    the six-rank global row.
    """
    total = 0
    for desc in descs:
        if desc.kind != ZEROFILL or int(desc.dst_rank) != int(dst_rank):
            continue
        ops.memset_async(int(desc.dst_ptr) + int(desc.dst_off), 0,
                         int(desc.nbytes), stream)
        total += int(desc.nbytes)
    if total:
        ops.synchronize(stream)
    return total


# ---------------------------------------------------------------------------
# The on-card lane.
# ---------------------------------------------------------------------------


@dataclass
class OnCardStats:
    card: int
    card_uuid: str
    mode: str
    bytes_moved: int = 0
    batches: int = 0
    elapsed_s: float = 0.0

    def line(self) -> str:
        """The second acceptance line of spec section 6/S4, plus ``batches``.

        ``hop_ms`` is THIS rank's half of the two hops, not the pair's total:
        the source's compaction and the destination's scatter run in two
        processes and neither can time the other without a clock they do not
        share.  E1 arm d predicts 11-18 ms for the whole diagonal.

        ``batches`` is NOT in the spec's token list and is added deliberately.
        ``bytes_mib`` is printed to two decimals, so anything under ~5 KiB
        prints ``0.00`` -- indistinguishable from a lane that moved NOTHING.
        On the real flip the diagonal is 10.28 GiB and the distinction never
        arises, which is exactly why it would never have been noticed: the
        six-process double, whose payloads are kilobytes, printed
        ``bytes_mib=0.00`` for a lane that had just moved every byte correctly.
        A line that cannot tell zero from small is not an acceptance, so the
        batch count sits beside the size and is a hard zero when nothing moved.

        IT IS ``batches``, NOT ``hops``, and the rename is the instrument-text
        rule applied to this file's own new field.  The field was added as
        ``hops`` and counts one per BATCH: the lane has exactly TWO hops by
        construction (compact into the bounce, scatter out of the peer's), so
        on the real flip ``hops=`` would have printed ~329 for a two-hop lane.
        A field whose name states something the number is not is the defect
        class this campaign keeps naming, and a field added to fix one
        blindness is the worst place to introduce it.  The structural two is
        not printed at all: a field that cannot be anything else is not a
        measurement (R2-10), and ``hop_ms`` already names what the two hops
        cost.
        """
        return (
            f"{ONCARD_LINE_PREFIX} card={self.card_uuid} mode={self.mode} "
            f"bytes_mib={self.bytes_moved / xr.MIB:.2f} batches={self.batches} "
            f"hop_ms={self.elapsed_s * 1e3:.3f}"
        )


@dataclass(frozen=True)
class OnCardSlotPlan:
    """What one card's diagonal costs at a chosen slot size, and the model.

    THE MODEL IS ONE LINE AND IT IS PRINTED WHEREVER THE NUMBER IS:
    ``hop_ms = batches x per_batch_ms``, ``batches = ceil(bytes / slot_bytes)``.
    Its coefficient is :data:`ONCARD_PER_BATCH_MS`, MEASURED consumer-side on
    the 5090 (S5-pre), and the measurement's own finding is what makes the
    model this shape: the per-batch cost did not move with the bytes per batch
    across an 8x sweep, so the batch COUNT is the only term worth choosing.

    ``fits`` grades the priced hop against spec 6/S4's ``hop_ms <= 20``.  It is
    a PREDICTION, and the boot compares it against the measured ``hop_ms=`` on
    the ``WEG2-XCHG-ONCARD`` line -- which is why both numbers are printed
    rather than one being trusted.  The extrapolation is honest about its
    range: 329 batches is 2.6x beyond the largest batch count S5-pre measured.
    """

    bytes_total: int
    slot_bytes: int
    slots: int
    batches: int
    per_batch_ms: float
    budget_ms: float

    @property
    def hop_ms(self) -> float:
        return self.batches * self.per_batch_ms

    @property
    def fits(self) -> bool:
        return self.hop_ms <= self.budget_ms

    @property
    def bounce_bytes(self) -> int:
        """The VRAM this choice costs on the producer's card."""
        return self.slots * self.slot_bytes

    def model(self) -> str:
        return (
            f"hop_ms = batches x per_batch_ms = {self.batches} x "
            f"{self.per_batch_ms:g} = {self.hop_ms:.1f} ms "
            f"(per_batch_ms MEASURED consumer-side on the 5090, S5-pre poll-rebind "
            f"arm at ONCARD_POLL_S={ONCARD_POLL_S * 1e6:g}us; batch count is the "
            f"only term that moved the cost across an 8x slot sweep)"
        )

    def tokens(self) -> str:
        return (
            f"oncard_slot_mib={self.slot_bytes / xr.MIB:.0f} "
            f"oncard_slots={self.slots} oncard_batches={self.batches} "
            f"oncard_hop_ms_priced={self.hop_ms:.1f} "
            f"oncard_hop_ms_budget={self.budget_ms:g} "
            f"oncard_priced_fits={'yes' if self.fits else 'no'}"
        )


def plan_oncard_slot_bytes(
    bytes_total: int,
    *,
    slots: int = ONCARD_SLOTS,
    budget_ms: float = ONCARD_HOP_BUDGET_MS,
    per_batch_ms: float = ONCARD_PER_BATCH_MS,
    floor_bytes: int = ONCARD_SLOT_BYTES,
    ceiling_bytes: int = ONCARD_SLOT_BYTES_MAX,
    align_bytes: int = xr.MIB,
) -> OnCardSlotPlan:
    """Choose the diagonal's slot size from the plan's own byte count.

    The diagonal's batch count is known at PLAN time -- it is
    ``ceil(oncard_bytes / slot_bytes)`` and nothing else -- so the slot size is
    a decision the plan can make instead of a constant the flip inherits.  With
    :data:`ONCARD_PER_BATCH_MS` measured, the smallest slot that clears the
    budget is ``bytes / floor(budget / per_batch)``, rounded UP to
    ``align_bytes`` and clamped into ``[floor_bytes, ceiling_bytes]``.

    IT MAY RETURN A PLAN THAT DOES NOT FIT, and that is deliberate: the
    ceiling is a VRAM residency term (:data:`ONCARD_SLOT_BYTES_MAX`), so on a
    diagonal large enough it is arithmetically impossible to clear 20 ms at
    this per-batch cost.  Returning ``fits=False`` with every term printed is
    a statement the boot can act on; silently exceeding the VRAM ceiling to
    make a timing budget green is the compensation-layer reflex.
    """
    total = int(bytes_total)
    slots = int(slots)
    if total <= 0:
        return OnCardSlotPlan(0, int(floor_bytes), slots, 0, float(per_batch_ms),
                              float(budget_ms))
    if per_batch_ms <= 0:
        raise ValueError(f"per_batch_ms must be positive, not {per_batch_ms!r}")
    allowed = int(float(budget_ms) // float(per_batch_ms))
    if allowed < 1:
        want = int(ceiling_bytes)
    else:
        want = -(-total // allowed)  # ceil
    want = -(-want // int(align_bytes)) * int(align_bytes)
    slot = max(int(floor_bytes), min(int(ceiling_bytes), want))
    batches = -(-total // slot)
    return OnCardSlotPlan(total, slot, slots, batches, float(per_batch_ms),
                          float(budget_ms))


class _OnCardWait:
    """Spin-then-sleep for the on-card row polls.  One per wait call.

    S5-pre measured the lane per-batch-bound and measured HALF of that cost to
    be this wait's sleep (see :data:`ONCARD_POLL_S`).  A bare
    ``time.sleep(poll)`` per iteration pays a full sleep granularity even when
    the peer's row lands microseconds later, which is the common case on a
    lock-step lane: the two ranks are co-located, the peer is inside a
    ``cudaStreamSynchronize`` of a copy this rank just watched it start.

    So: spin (a bare re-read, no syscall) for :data:`ONCARD_SPIN_BUDGET_S` --
    the measured floor of the round trip -- and only then sleep
    :data:`ONCARD_POLL_S` per iteration.  BOTH BOUNDS ARE REQUIRED.  A spin
    with no time bound burns a core for a peer that died; a spin with no
    iteration bound spins an unmeasured number of times on a machine where a
    read is cheap.  The overall wait is still bounded by the fence budget at
    the call site -- this class chooses only HOW to wait, never HOW LONG.

    ``sleep``/``monotonic`` are injectable because the property under test is
    "did it sleep yet", which is unobservable through a real clock without
    making a timing assertion out of it.
    """

    __slots__ = ("_iters", "_budget", "_poll", "_sleep", "_monotonic",
                 "_started", "_spun", "_slept")

    def __init__(self, *, spin_iters: int = ONCARD_SPIN_ITERS,
                 spin_budget_s: float = ONCARD_SPIN_BUDGET_S,
                 poll_s: float = ONCARD_POLL_S,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic):
        self._iters = int(spin_iters)
        self._budget = float(spin_budget_s)
        self._poll = float(poll_s)
        self._sleep = sleep
        self._monotonic = monotonic
        self._started = monotonic()
        self._spun = 0
        self._slept = 0

    @property
    def spins(self) -> int:
        return self._spun

    @property
    def sleeps(self) -> int:
        return self._slept

    def wait(self) -> None:
        if (self._spun < self._iters
                and self._monotonic() - self._started < self._budget):
            self._spun += 1
            return
        self._slept += 1
        self._sleep(self._poll)


def _oncard_row_off(area_off: int, row: int, slot: int = 0) -> int:
    if not 0 <= int(row) < xr.N_RANKS:
        raise ValueError(f"row must be 0..{xr.N_RANKS - 1}, not {row!r}")
    if not 0 <= int(slot) < ONCARD_SLOTS_MAX:
        raise ValueError(f"slot must be 0..{ONCARD_SLOTS_MAX - 1}, not {slot!r}")
    index = int(row) * ONCARD_SLOTS_MAX + int(slot)
    return area_off + index * DIR_ONCARD_ROW_BYTES


def write_oncard_row(region: xr.XchgRegion, area_off: int, row: int, *,
                     seq: int, nbytes: int, slot_bytes: int, wave: int,
                     state: int, slot: int = 0, checksum: int = 0) -> None:
    """Publish one side's on-card handshake row.  ONE WRITER PER ADDRESS.

    Producer rows and consumer rows are separate arrays, so no word here has two
    writers and no atomic is needed.  The seal is what makes a half-written row
    not a signal: a reader that accepted an unsealed row would take a stale
    ``nbytes`` beside a fresh ``seq``, which is the torn-row defect S3 already
    pinned with a test one area over.

    ``slot`` is not decoration: a row is per rank PER SLOT (see
    :data:`DIR_ONCARD_ROWS`), so batch k's description survives until the
    producer is allowed to reuse slot ``k % ONCARD_SLOTS`` -- which the double
    buffer forbids until the consumer has drained it.

    ``wave`` and ``slot_bytes`` are REQUIRED, never defaulted: both are stamps
    a reader compares against its own, and a default would let a caller publish
    a row that agrees with everybody by accident.  See
    :data:`ONCARD_ROW_STRUCT` for the two defects each of them closes.
    """
    region._require_flip(f"write_oncard_row row={row} slot={slot}")
    view = region.dir_view()
    off = _oncard_row_off(area_off, row, slot)
    payload = ONCARD_ROW_STRUCT.pack(int(seq), int(nbytes), int(slot_bytes),
                                     int(wave), region.epoch_hash,
                                     os.getpid(), int(state), int(checksum))
    addr = ctypes.addressof(view)
    ctypes.memmove(addr + off, payload, len(payload))
    ctypes.memmove(addr + off + ONCARD_SEAL_OFF, struct.pack("<Q", _seal(payload)), 8)


def read_oncard_row(region: xr.XchgRegion, area_off: int, row: int,
                    slot: int = 0) -> Dict[str, int]:
    view = region.dir_view()
    off = _oncard_row_off(area_off, row, slot)
    addr = ctypes.addressof(view)
    payload = ctypes.string_at(addr + off, ONCARD_ROW_STRUCT.size)
    seal = struct.unpack("<Q", ctypes.string_at(addr + off + ONCARD_SEAL_OFF, 8))[0]
    seq, nbytes, slot_bytes, wave, eh, pid, state, checksum = \
        ONCARD_ROW_STRUCT.unpack(payload)
    return {
        "seq": seq, "bytes": nbytes, "slot_bytes": slot_bytes, "wave": wave,
        "epoch_hash": eh, "pid": pid, "checksum": checksum,
        "state": state, "sealed": int(seal == _seal(payload)),
    }


def write_oncard_release(region: xr.XchgRegion, row: int, *, wave: int) -> None:
    """"My mapping of the peer's bounce is gone."  Written by the CONSUMER.

    The one signal that makes the producer's ``cudaFree`` legal.  It is written
    on EVERY path out of the consumer's diagonal, the raising ones included,
    because a producer waiting for it must not be held by the consumer's
    failure -- the failure it should hear about is the peer's FAILED row, which
    arrives first and is what actually stops it.
    """
    write_oncard_row(region, DIR_ONCARD_REL_OFF, row, slot=0, seq=0, nbytes=0,
                     slot_bytes=0, wave=wave, state=ONCARD_STATE_DONE)


def mark_oncard_failed(region: xr.XchgRegion, area_off: int, row: int, *,
                       wave: int) -> None:
    """Vote this lane FAILED on every slot, so the peer stops NOW.

    :func:`_await_oncard` checks FAILED before it checks the sequence, so this
    turns a peer's 120 s fence-budget wait into an immediate refusal that names
    the death rather than the wait.  Best-effort by construction: it runs on an
    unwind path, and a region with no flip bound (the one case where the write
    itself refuses) is a rank that never moved a byte anyway.
    """
    for slot in range(ONCARD_SLOTS):
        try:
            write_oncard_row(region, area_off, row, slot=slot, seq=0, nbytes=0,
                             slot_bytes=0, wave=wave,
                             state=ONCARD_STATE_FAILED)
        except Exception:  # noqa: BLE001 -- an unwind path must not raise again
            return


def publish_ipc_handle(region: xr.XchgRegion, row: int, handle: bytes) -> None:
    if len(handle) != CUDA_IPC_HANDLE_SIZE:
        raise ValueError(
            f"an IPC handle is exactly {CUDA_IPC_HANDLE_SIZE} bytes, got "
            f"{len(handle)} -- see the c_char truncation defect S0 caught")
    view = region.dir_view()
    off = DIR_HANDLE_OFF + int(row) * CUDA_IPC_HANDLE_SIZE
    if not 0 <= int(row) < xr.N_RANKS:
        raise ValueError(f"row must be 0..{xr.N_RANKS - 1}, not {row!r}")
    ctypes.memmove(ctypes.addressof(view) + off, handle, CUDA_IPC_HANDLE_SIZE)


def read_ipc_handle(region: xr.XchgRegion, row: int) -> bytes:
    if not 0 <= int(row) < xr.N_RANKS:
        raise ValueError(f"row must be 0..{xr.N_RANKS - 1}, not {row!r}")
    view = region.dir_view()
    off = DIR_HANDLE_OFF + int(row) * CUDA_IPC_HANDLE_SIZE
    return ctypes.string_at(ctypes.addressof(view) + off, CUDA_IPC_HANDLE_SIZE)


def arm_oncard_lane(
    *,
    card_uuid: str,
    probe: Callable[[], Tuple[bool, str]],
    requested: str = "",
    log: Callable[[str], None],
) -> str:
    """Decide the on-card lane's mode ONCE, at the launcher (spec 3.7, W56).

    ``probe`` is the S0 probe as a subprocess: it returns ``(ok, reason)``.  A
    ``False`` is **W56** -- logged by name, with the degrade named and the cost
    named -- and the boot then runs the on-card share through the staging region
    for its whole life.  There is no per-flip and no per-lane retry (R2-1).

    An explicit ``--weg2-xchg-oncard host`` skips the probe entirely: the user
    asked for the degrade, and probing anyway would let a green probe silently
    override the request.  It still logs the W56 line, because a boot running
    the degrade must say so whichever way it got there.

    An explicit ``--weg2-xchg-oncard ipc`` whose probe FAILS is the one case
    that RAISES :class:`Weg2XchgOnCardUnavailable` instead of degrading.  Left
    to the default the degrade is spec 3.7's own answer and is logged; asked
    for by name, silently handing back ``host`` would run a boot whose operator
    believes the lane is on -- and every wall claim that follows would be about
    a lane that is not there.  This is also what gives the class a runtime use:
    a refusal that is only ever a string is a refusal nobody can catch.

    TODO(S6): the launcher's ``prepare_weight_exchange`` is where this belongs
    on the boot path, beside W55.  It is unwired here on purpose --
    ``XCHG_RANK_BEHAVIOUR_WIRED`` on the ``WEG2-XCHG-ARMED`` line is still
    ``no``, and arming a lane whose consumer does not exist would be the
    unarmed-gate-reading-as-a-passed-one defect this design keeps naming.
    """
    requested = (requested or os.environ.get(ENV_ONCARD_MODE, "") or "").strip()
    if requested == ONCARD_MODE_HOST:
        log(oncard_unavailable_message(
            card=card_uuid, reason="requested=host", mode=ONCARD_MODE_HOST))
        return ONCARD_MODE_HOST
    if requested not in ("", ONCARD_MODE_IPC):
        raise ValueError(
            f"{ENV_ONCARD_MODE} must be {ONCARD_MODE_IPC!r} or "
            f"{ONCARD_MODE_HOST!r}, not {requested!r}")
    try:
        ok, reason = probe()
    except Exception as exc:  # noqa: BLE001 -- a probe that raises is a NO
        ok, reason = False, f"{type(exc).__name__}: {exc}"
    if ok:
        return ONCARD_MODE_IPC
    reason = reason or "cudaIpc unavailable"
    if requested == ONCARD_MODE_IPC:
        message = oncard_refused_message(card=card_uuid, reason=reason)
        log(message)
        raise Weg2XchgOnCardUnavailable(message)
    log(oncard_unavailable_message(
        card=card_uuid, reason=reason, mode=ONCARD_MODE_HOST))
    return ONCARD_MODE_HOST


class OnCardBounce:
    """The raw ``cudaMalloc`` bounce buffer and its IPC handle.

    Allocated by the PRODUCER of a card's on-card hop and imported by the
    co-located consumer (spec section 1.3 step 16: *"S D2Ds its VRAM -> its IPC
    bounce slot ... W D2Ds the imported peer pointer -> its VRAM"*).  Raw, so
    the base pointer is a TRUE base rather than an offset into a torch segment,
    and outside every TMS region, so it never enters the saver's census -- both
    asserted on the metal by S0 (``tms_census_hit=0``).
    """

    def __init__(self, ops: DeviceOps, device: int, slots: int = ONCARD_SLOTS,
                 slot_bytes: int = ONCARD_SLOT_BYTES):
        self.ops = ops
        self.slots = int(slots)
        self.slot_bytes = int(slot_bytes)
        self.nbytes = self.slots * self.slot_bytes
        self.ptr = ops.raw_malloc(int(device), self.nbytes)
        self.handle = ops.ipc_get_handle(self.ptr)

    def slot_address(self, slot: int) -> int:
        return self.ptr + (int(slot) % self.slots) * self.slot_bytes

    def close(self) -> None:
        self.ops.raw_free(self.ptr)


def oncard_host_path(boot_nonce: str, card: int, shm_root: str = xr.SHM_ROOT) -> str:
    """The degrade lane's per-card bounce file.

    A SEPARATE FILE, not a seventh staging pair.  :data:`xr.CROSS_PAIRS` has no
    diagonal by construction -- S3 excluded it because the on-card lane was
    never meant to take a slot -- so the ``host`` degrade cannot borrow one, and
    widening S3's layout to seven pairs would cost 64 MiB of pinned host on
    EVERY boot to serve an arm that S0 measured as not taken.  The file is
    created only when the degrade is armed, and it joins the same residue sweep
    as the region because it lives under the same boot directory.

    OPEN ITEM, named rather than hidden: on the degrade arm this adds
    ``3 x 2 x 32 MiB = 192 MiB`` of host that spec 0.2's ledger term (0.38 GiB
    staging) does NOT carry.  Section 3.7 prices the degrade in WALL (+0.49 s
    on the x4 card) and not in host.  The saving is still ~34.5 GiB, so the arm
    remains far inside the bound -- but the number belongs in the record, not
    in a comment nobody reads, and it is repeated in the S4 record block.
    """
    return os.path.join(xr.region_dir(boot_nonce, shm_root), f"oncard-{int(card)}.bin")


class HostBounce:
    """The ``host`` degrade's bounce: a small shm file instead of device VRAM.

    Same geometry, same handshake rows, same batcher, same W54 comparison as
    :class:`OnCardBounce` -- only the storage differs, which is the whole point
    of routing the degrade through this class rather than through a second copy
    of the loop.  Registered with ``cudaHostRegister`` for the same measured
    7.5 % / 3.3 % reason the staging region is.
    """

    def __init__(self, ops: DeviceOps, boot_nonce: str, card: int, *,
                 create: bool, slots: int = ONCARD_SLOTS,
                 slot_bytes: int = ONCARD_SLOT_BYTES,
                 shm_root: str = xr.SHM_ROOT):
        import mmap as _mmap

        self.ops = ops
        self.slots = int(slots)
        self.slot_bytes = int(slot_bytes)
        self.nbytes = self.slots * self.slot_bytes
        self.path = oncard_host_path(boot_nonce, card, shm_root)
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
        self.handle = b""
        self._registered = False
        try:
            ops.host_register(self.ptr, self.nbytes, CUDA_HOST_REGISTER_PORTABLE)
            self._registered = True
        except Exception:  # noqa: BLE001 -- a 7.5 % regression, not a refusal
            self._registered = False

    def slot_address(self, slot: int) -> int:
        return self.ptr + (int(slot) % self.slots) * self.slot_bytes

    def close(self) -> None:
        if self._registered:
            try:
                self.ops.host_unregister(self.ptr)
            finally:
                self._registered = False
        self._mm.close()
        os.close(self._fd)


def run_oncard_producer(
    region: xr.XchgRegion,
    ops: DeviceOps,
    stream: int,
    bounce: OnCardBounce,
    *,
    row: int,
    peer_row: int,
    wave: int,
    descs: Sequence[object],
    stats: OnCardStats,
    budget_s: Optional[float] = None,
    checksum_bytes: Optional[Callable[[int, int], int]] = None,
) -> OnCardStats:
    """Hop 1: this rank's VRAM -> its own bounce slot, batch by batch.

    The row is written only AFTER the sync, exactly as ``bytes_filled`` is on
    the staging lane and for the same reason: the row is this lane's "these
    bytes have landed" signal, and a row written before the sync would mean
    "these bytes were issued".

    IT ENDS WITH A TERMINAL DRAIN, and that is a lifetime property rather than
    a tidiness one -- see :data:`DIR_ONCARD_REL_OFF`.  Without it the producer
    returns while its consumer is still issuing D2D out of the last
    ``ONCARD_SLOTS`` batches, and the caller's ``finally`` frees the buffer
    underneath them.
    """
    budget = xr.fence_budget_s() if budget_s is None else float(budget_s)
    batches = batch_descs(descs, bounce.slot_bytes)
    started = time.perf_counter()
    for batch in batches:
        slot = batch.seq % bounce.slots
        # The double buffer's only rule: slot s may be refilled once the
        # consumer has released the batch that used it last, which is
        # `seq - slots`.  Everything else about this lane's safety follows.
        _await_oncard(region, DIR_ONCARD_CONS_OFF, peer_row,
                      batch.seq - bounce.slots, budget, what="drain", row=row,
                      slot=slot, wave=wave)
        base = bounce.slot_address(batch.seq)
        for piece in batch.pieces:
            desc = descs[piece.desc_index]
            src_ptr = int(desc.src_ptr) + piece.src_off
            if piece.kind == FLAT:
                ops.memcpy_async(base + piece.slot_off, src_ptr, piece.nbytes,
                                 stream)
            else:
                ops.memcpy2d_async(base + piece.slot_off, piece.run_bytes,
                                   src_ptr, piece.spitch,
                                   piece.run_bytes, piece.rows, stream)
        ops.synchronize(stream)
        write_oncard_row(region, DIR_ONCARD_PROD_OFF, row, slot=slot,
                         seq=batch.seq, nbytes=batch.total_bytes,
                         slot_bytes=bounce.slot_bytes, wave=wave,
                         state=ONCARD_STATE_READY,
                         checksum=batch.checksum(checksum_bytes, base))
        stats.bytes_moved += batch.total_bytes
        stats.batches += 1
    # This rank's OWN half of the two hops, measured before the terminal wait:
    # `hop_ms` must not silently absorb how long the peer took, or a slow
    # consumer would read as a slow producer on the acceptance line.
    stats.elapsed_s = time.perf_counter() - started
    if batches:
        last = batches[-1].seq
        for seq in range(max(0, last - bounce.slots + 1), last + 1):
            _await_oncard(region, DIR_ONCARD_CONS_OFF, peer_row, seq, budget,
                          what="drain-final", row=row,
                          slot=seq % bounce.slots, wave=wave)
    return stats


def run_oncard_consumer(
    region: xr.XchgRegion,
    ops: DeviceOps,
    stream: int,
    peer_ptr: int,
    *,
    row: int,
    peer_row: int,
    wave: int,
    descs: Sequence[object],
    stats: OnCardStats,
    slots: int = ONCARD_SLOTS,
    slot_bytes: int = ONCARD_SLOT_BYTES,
    budget_s: Optional[float] = None,
    checksum_bytes: Optional[Callable[[int, int], int]] = None,
    on_checksum: Optional[Callable[[ChecksumReport], None]] = None,
) -> OnCardStats:
    """Hop 2: the imported peer bounce -> this rank's VRAM.

    The same W54 comparison as the staging lane, for the same reason: the
    producer's published ``bytes`` is a claim and ``batch.total_bytes`` is this
    rank's own derivation.  A hop that copies at the producer's claim takes the
    previous batch's residue with it and raises nothing.
    """
    budget = xr.fence_budget_s() if budget_s is None else float(budget_s)
    batches = batch_descs(descs, slot_bytes)
    started = time.perf_counter()
    for batch in batches:
        slot = batch.seq % int(slots)
        # EXACT, not >=: this row describes slot ``slot``'s current batch, and
        # the producer cannot have moved past it (it would have to reuse the
        # slot, which needs this consumer's release).  A >= here is what let
        # batch 1's byte count answer batch 0's question and raise W54 against
        # a healthy producer.
        got = _await_oncard(region, DIR_ONCARD_PROD_OFF, peer_row, batch.seq,
                            budget, what="fill", row=row, slot=slot,
                            wave=wave, exact=True)
        # The producer's batcher geometry, published rather than assumed.  Two
        # co-located ranks reading the same plan with different slot sizes
        # agree on every payload below the smaller one and diverge silently
        # above it; on this lane the row can carry the number, so it does.
        if got["slot_bytes"] != int(slot_bytes):
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree oncard row={row} peer_row={peer_row} "
                f"slot={slot} seq={batch.seq} epoch={region.epoch} wave={wave}: "
                f"the producer batched at slot_bytes={got['slot_bytes']} and "
                f"this consumer at {int(slot_bytes)} -- the two sides would "
                f"place bytes at different offsets from the same plan; NO copy "
                f"was issued"
            )
        if got["bytes"] != batch.total_bytes:
            raise Weg2XchgShortPiece(short_piece_message(
                lane="oncard", pair=-1, slot=slot, seq=batch.seq,
                expected=batch.total_bytes, filled=got["bytes"],
                seq_filled=got["seq"],
                src=peer_row, dst=row, epoch=region.epoch,
                producer_pid=got["pid"],
            ))
        base = peer_ptr + slot * int(slot_bytes)
        _report_checksum(on_checksum, checksum_bytes, batch, base,
                         lane="oncard", pair=-1, slot=slot,
                         published=got["checksum"])
        for piece in batch.pieces:
            desc = descs[piece.desc_index]
            dst_ptr = int(desc.dst_ptr) + piece.dst_off
            if piece.kind == FLAT:
                ops.memcpy_async(dst_ptr, base + piece.slot_off, piece.nbytes,
                                 stream)
            else:
                ops.memcpy2d_async(dst_ptr, piece.dpitch,
                                   base + piece.slot_off, piece.run_bytes,
                                   piece.run_bytes, piece.rows, stream)
        ops.synchronize(stream)
        write_oncard_row(region, DIR_ONCARD_CONS_OFF, row, slot=slot,
                         seq=batch.seq, nbytes=batch.total_bytes,
                         slot_bytes=int(slot_bytes), wave=wave,
                         state=ONCARD_STATE_DONE)
        stats.bytes_moved += batch.total_bytes
        stats.batches += 1
    stats.elapsed_s = time.perf_counter() - started
    return stats


def _await_oncard(region: xr.XchgRegion, area_off: int, peer_row: int, seq: int,
                  budget_s: float, *, what: str, row: int, wave: int,
                  slot: int = 0, exact: bool = False,
                  wait: Optional["_OnCardWait"] = None) -> Dict[str, int]:
    """Bounded poll for a peer's on-card row to reach ``seq``.

    Bounded by ``WEG2_GROUP_FENCE_BUDGET_S`` like every other wait in this
    design -- no second timeout constant (section 3.2).  A row counts only if it
    is SEALED, carries THIS flip's epoch hash, carries THIS WAVE, and has
    reached the sequence asked for.  Dropping the epoch check is how a gate
    passes while a rank is absent: the previous flip left the row at its last
    sequence, so the first poll of the next flip would return immediately with
    nobody having filled anything -- the wave gate's own stop-loss, one lane
    over.

    **AND THE WAVE CHECK IS THE SAME PROPERTY AT THE SMALLER SCALE.**
    MEASURED-BY-REVIEW DEFECT (S4 refuter, 2026-09-09, "must_fix"): the epoch
    hash changes per FLIP, but :func:`run_leg` runs per WAVE and each wave
    restarts ``seq`` at 0.  Wave 2's producer therefore found wave 1's leftover
    consumer row, at the same epoch and at a much larger sequence, satisfied
    its ``>=`` drain wait instantly, and refilled a slot nobody had drained.
    The FILL direction had the mirror image: an on-card share of two batches or
    fewer matches wave 1's exact row and either raises W54 against a healthy
    producer or copies out of an unfilled bounce.  Epoch alone is not identity
    when the unit of work is smaller than a flip.

    ``exact`` distinguishes the two questions this poll answers.  The FILL
    direction asks "is slot s carrying batch k", and must be exact: a ``>=``
    accepts a later batch's byte count as an answer about this one, which is
    how a healthy producer earned a W54 on the first green run of this suite.
    The DRAIN direction asks "has the consumer got at least as far as k", and
    ``>=`` is the honest reading there -- WITHIN a wave, which is what the wave
    stamp now guarantees.

    A negative ``seq`` is the base case of the double buffer -- there is no
    batch ``-1`` to wait for -- and returns without reading anything.
    """
    if seq < 0:
        return {"seq": seq, "bytes": 0, "slot_bytes": 0, "wave": int(wave),
                "state": ONCARD_STATE_DONE, "pid": 0, "checksum": 0,
                "epoch_hash": region.epoch_hash, "sealed": 1}
    started = time.monotonic()
    waiter = _OnCardWait() if wait is None else wait
    while True:
        got = read_oncard_row(region, area_off, peer_row, slot)
        fresh = (got["sealed"] and got["epoch_hash"] == region.epoch_hash
                 and got["wave"] == int(wave)
                 and got["state"] not in (ONCARD_STATE_IDLE, ONCARD_STATE_ARMED))
        if fresh and got["state"] == ONCARD_STATE_FAILED:
            raise Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout oncard row={row} peer_row={peer_row} "
                f"slot={slot} epoch={region.epoch} wave={wave} seq={seq} "
                f"what={what} -- the peer voted FAILED on this lane"
            )
        if fresh and (got["seq"] == seq if exact else got["seq"] >= seq):
            return got
        if time.monotonic() - started >= budget_s:
            raise Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout oncard row={row} peer_row={peer_row} "
                f"slot={slot} epoch={region.epoch} wave={wave} seq={seq} "
                f"what={what} exact={'yes' if exact else 'no'} "
                f"budget_s={budget_s} waited_s={time.monotonic() - started:.3f} "
                f"peer_seq={got['seq']} peer_wave={got['wave']} "
                f"peer_state={got['state']} "
                f"peer_sealed={got['sealed']} peer_pid={got['pid']} "
                f"alive_in_proc={'yes' if _pid_alive(got['pid'], '/proc') else 'no'} "
                f"(denominator: the peer's on-card row for THIS slot carrying "
                f"epoch_hash={region.epoch_hash:#x} and wave={wave})"
            )
        waiter.wait()


def _await_handle(region: xr.XchgRegion, peer_row: int, budget_s: float,
                  *, row: int, wave: int) -> Dict[str, int]:
    """Wait for the co-located producer to publish its bounce IPC handle.

    Separate from :func:`_await_oncard` because it waits on a DIFFERENT event:
    the producer arms (``ONCARD_STATE_ARMED``, ``seq = -1``, slot 0) once,
    before batch 0, and the consumer must import the handle before the first
    batch exists.  Folding the two would make batch 0's wait also the handle's
    wait, and the consumer would then open a handle it had never been told was
    there.

    THE WAVE STAMP IS WHY IT IS SAFE TO READ THE HANDLE AFTER THIS RETURNS.
    Every wave allocates a NEW bounce and publishes a NEW 64-byte handle over
    the old one, and :func:`publish_ipc_handle` is an unsealed ``memmove`` with
    no publication barrier of its own.  Accepting any non-IDLE row at this
    epoch -- which is what this did -- let waves 2 and 3 skip the wait entirely
    and read wave 1's handle, or a torn mix of two.  The row IS the barrier:
    it is sealed, it is written after the handle, and it now carries the wave.
    """
    started = time.monotonic()
    waiter = _OnCardWait()
    while True:
        got = read_oncard_row(region, DIR_ONCARD_PROD_OFF, peer_row, 0)
        if (got["sealed"] and got["epoch_hash"] == region.epoch_hash
                and got["wave"] == int(wave)
                and got["state"] != ONCARD_STATE_IDLE):
            return got
        if time.monotonic() - started >= budget_s:
            raise Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout oncard row={row} peer_row={peer_row} "
                f"epoch={region.epoch} wave={wave} what=handle "
                f"budget_s={budget_s} "
                f"waited_s={time.monotonic() - started:.3f} "
                f"peer_state={got['state']} peer_sealed={got['sealed']} "
                f"peer_wave={got['wave']} peer_pid={got['pid']} "
                f"alive_in_proc={'yes' if _pid_alive(got['pid'], '/proc') else 'no'} "
                f"-- the co-located source never published a bounce handle"
            )
        waiter.wait()


def _await_release(region: xr.XchgRegion, peer_row: int, budget_s: float,
                   *, row: int, wave: int) -> Dict[str, int]:
    """Wait for the co-located consumer to have CLOSED the imported pointer.

    The second half of the lifetime handshake (:data:`DIR_ONCARD_REL_OFF`).
    The terminal drain proves the peer's last copy has LANDED; this proves its
    MAPPING is gone, which is the condition CUDA puts on the exporter's
    ``cudaFree``.  Two questions, two signals: a drain alone still frees under
    an open mapping.
    """
    return _await_oncard(region, DIR_ONCARD_REL_OFF, peer_row, 0, budget_s,
                         what="release", row=row, slot=0, wave=wave)


# ---------------------------------------------------------------------------
# One rank's whole leg: 2 cross-pair threads + 1 on-card thread.
# (Spec 6/S4's "2 producer + 2 consumer + 1 on-card" is the per-CARD total
# across the two co-located processes -- see run_leg's docstring.)
# ---------------------------------------------------------------------------


@dataclass
class LegResult:
    pairs: List[PairStats] = field(default_factory=list)
    oncard: Optional[OnCardStats] = None
    zerofill_bytes: int = 0
    lines: List[str] = field(default_factory=list)


def run_leg(
    region: xr.XchgRegion,
    sems: SemSet,
    ops: DeviceOps,
    *,
    row: int,
    rank: int,
    device: int,
    card_uuid: str,
    uuid_of_card: Sequence[str],
    descs: Sequence[object],
    is_source: bool,
    oncard_mode: str,
    peer_row: int,
    wave: int,
    log: Callable[[str], None],
    vote_failure: Callable[[BaseException], None],
    budget_s: Optional[float] = None,
    slot_bytes: int = xr.SLOT_BYTES,
    oncard_slot_bytes: Optional[int] = None,
    oncard_slots: int = ONCARD_SLOTS,
    checksum_bytes: Optional[Callable[[int, int], int]] = None,
    on_checksum: Optional[Callable[[ChecksumReport], None]] = None,
) -> LegResult:
    """Move this rank's whole share of one wave.

    THREE THREADS PER RANK, five per CARD.  A rank in a wake is either a source
    or a destination (``is_source`` selects one direction of
    :data:`xr.CROSS_PAIRS`, spec 1.3), so it runs its TWO cross-pair threads --
    outbound if it is a source, inbound if it is a destination -- plus ONE
    on-card thread, each with its own stream; a destination adds a fourth
    stream afterwards for the ZEROFILL memsets.  Spec section 6/S4's "2
    producer + 2 consumer + 1 on-card" is the per-CARD total across the two
    co-located processes, not one process's thread count.  (The docstring said
    five per rank until the S4 review counted them.)

    Separate streams rather than one because the two PCIe directions and the
    diagonal are independent engines on this hardware; serialising them onto
    one stream makes the leg the SUM of the rates instead of their MAX.

    ``wave`` IS REQUIRED.  It is the unit this function is called in, and every
    on-card handshake row carries it: without it wave 2 reads wave 1's rows and
    wave 1's IPC handle at the same epoch hash (see :func:`_await_oncard`).

    ``vote_failure`` IS REQUIRED, NOT DEFAULTED, and it is a deadlock property
    rather than a convenience.  A thread that raises leaves its slot FILLING
    with ``full`` unposted; the peer would then block to the fence budget and
    report a timeout naming the WAIT rather than the DEATH.  The caller votes
    ``ok=False`` into its gate row so every peer refuses at once.  A default of
    ``None`` would make that opt-in, and an opt-in unwind is the weg2rg2
    hold-and-wait one layer up.

    ``rank`` is the GROUP-LOCAL rank (also the positional card index, and what
    ``XchgDesc.src_rank``/``dst_rank`` carry); ``row`` is the six-rank global
    gate row; ``device`` is the CUDA ordinal this process sees.  Three names
    because they are three numbers, and the boot forms in this tree have made
    every pair of them differ at least once.

    ``oncard_slot_bytes`` DEFAULTS TO ``slot_bytes``, and that default is a
    measured correction rather than a convenience: the first version took
    ``slot_bytes`` for the cross lane and left the on-card bounce on the module
    default, so a caller that set one knob got two different batch sizes and no
    line said so.  A knob that applies to half of what its name covers is worse
    than no knob.

    TODO(S6): nothing calls this yet.  S6 owns the RPC handler that binds the
    flip, runs Gate 0, resumes the destination's tags, calls this per wave and
    then closes ``wave_gate``.  ``test_run_leg_interface_is_what_s6_must_call``
    pins the signature so the seam cannot drift while it is unwired.
    """
    budget = xr.fence_budget_s() if budget_s is None else float(budget_s)
    # Both knobs bounded BEFORE a thread starts, and each against the storage
    # it actually writes into: the cross knob against this region's own header,
    # the diagonal knob against the bounce geometry this module owns.  See
    # `require_slot_bytes` for the defect.
    slot_bytes = require_slot_bytes(region.header()["slot_bytes"], slot_bytes,
                                    what=f"run_leg row={row} lane=cross")
    diag_bytes = require_slot_bytes(
        ONCARD_SLOT_BYTES_MAX,
        slot_bytes if oncard_slot_bytes is None else oncard_slot_bytes,
        what=f"run_leg row={row} lane=oncard")
    # The pipeline DEPTH is bounded by the row area, which is sized once from
    # ONCARD_SLOTS_MAX and can therefore not follow an argument.  Refused here,
    # before a thread exists, for the same reason as the slot size: a depth the
    # rows cannot address writes rank r's slot k over rank r+1's slot 0, and
    # both sides would compute the same wrong offset without disagreeing.
    diag_slots = require_oncard_slots(oncard_slots,
                                      what=f"run_leg row={row} lane=oncard")
    result = LegResult()
    errors: List[BaseException] = []
    lock = threading.Lock()

    def guarded(fn: Callable[[], None]) -> Callable[[], None]:
        def inner() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 -- re-raised below
                with lock:
                    errors.append(exc)
        return inner

    threads: List[threading.Thread] = []
    pair_stats: List[PairStats] = []
    cross = [d for d in descs if d.kind != ZEROFILL and not _on_card(d)]
    on_card = [d for d in descs
               if d.kind != ZEROFILL and _on_card(d) and int(d.src_rank) == rank]

    for pair, (src_card, dst_card) in enumerate(xr.CROSS_PAIRS):
        if (src_card if is_source else dst_card) != rank:
            continue
        stats = PairStats(src_card, dst_card,
                          uuid_of_card[src_card], uuid_of_card[dst_card])
        pair_stats.append(stats)
        sub = pair_descs(cross, src_card, dst_card)

        def one(pair=pair, stats=stats, sub=sub) -> None:
            stream = ops.create_stream(device)
            try:
                if is_source:
                    run_producer_pair(region, sems, ops, stream, pair=pair,
                                      descs=sub, stats=stats, budget_s=budget,
                                      slot_bytes=slot_bytes,
                                      checksum_bytes=checksum_bytes)
                else:
                    run_consumer_pair(region, sems, ops, stream, pair=pair,
                                      descs=sub, stats=stats, budget_s=budget,
                                      slot_bytes=slot_bytes,
                                      checksum_bytes=checksum_bytes,
                                      on_checksum=on_checksum)
            finally:
                ops.destroy_stream(stream)

        threads.append(threading.Thread(target=guarded(one),
                                        name=f"weg2-xchg-pair{pair}"))

    oncard_stats = OnCardStats(rank, card_uuid, oncard_mode)

    def diagonal() -> None:
        if not on_card:
            return
        ipc = oncard_mode == ONCARD_MODE_IPC
        stream = ops.create_stream(device)
        bounce = None
        peer_ptr = 0
        # An EXPORTED bounce may not be freed until the importer has closed it
        # -- CUDA's own rule and the use-after-free the S4 review found.  Every
        # other case may free at once: a HostBounce is this process's own
        # mapping and the consumer holds an independent one of the same file.
        may_free = not (is_source and ipc)
        try:
            if is_source:
                bounce = (OnCardBounce(ops, device, slots=diag_slots,
                                       slot_bytes=diag_bytes) if ipc
                          else HostBounce(ops, region.boot_nonce, rank,
                                          create=True, slots=diag_slots,
                                          slot_bytes=diag_bytes))
                if ipc:
                    publish_ipc_handle(region, row, bounce.handle)
                # ARMED, seq -1, slot 0: "the bounce is there, no batch yet".
                # A consumer that waited on batch 0 for the handle would open a
                # buffer it had never been told existed.  Written AFTER the
                # handle, sealed, and carrying this wave: it is the handle's
                # publication barrier as well as its announcement.
                write_oncard_row(region, DIR_ONCARD_PROD_OFF, row, slot=0,
                                 seq=-1, nbytes=0, slot_bytes=diag_bytes,
                                 wave=wave, state=ONCARD_STATE_ARMED)
                run_oncard_producer(region, ops, stream, bounce, row=row,
                                    peer_row=peer_row, wave=wave,
                                    descs=on_card, stats=oncard_stats,
                                    budget_s=budget,
                                    checksum_bytes=checksum_bytes)
                if ipc:
                    _await_release(region, peer_row, budget, row=row, wave=wave)
                    may_free = True
            else:
                armed = _await_handle(region, peer_row, budget, row=row,
                                      wave=wave)
                if armed["slot_bytes"] != diag_bytes:
                    raise Weg2XchgPlanDisagree(
                        f"W52 Weg2XchgPlanDisagree oncard row={row} "
                        f"peer_row={peer_row} epoch={region.epoch} wave={wave}: "
                        f"the co-located source armed a bounce of "
                        f"slot_bytes={armed['slot_bytes']} and this rank "
                        f"expects {diag_bytes} -- the two sides would read the "
                        f"same bounce at different slot addresses"
                    )
                if ipc:
                    peer_ptr = ops.ipc_open_handle(read_ipc_handle(region, peer_row))
                else:
                    bounce = HostBounce(ops, region.boot_nonce, rank,
                                        create=False, slots=diag_slots,
                                        slot_bytes=diag_bytes)
                    peer_ptr = bounce.ptr
                run_oncard_consumer(region, ops, stream, peer_ptr, row=row,
                                    peer_row=peer_row, wave=wave,
                                    descs=on_card, stats=oncard_stats,
                                    budget_s=budget, slots=diag_slots,
                                    slot_bytes=diag_bytes,
                                    checksum_bytes=checksum_bytes,
                                    on_checksum=on_checksum)
        except BaseException:
            # Vote this lane down where the peer is looking, so its wait ends
            # NOW and names the death instead of the fence budget.
            mark_oncard_failed(
                region,
                DIR_ONCARD_PROD_OFF if is_source else DIR_ONCARD_CONS_OFF,
                row, wave=wave)
            raise
        finally:
            if ipc and peer_ptr:
                ops.ipc_close_handle(peer_ptr)
            if not is_source:
                # On EVERY path, the raising ones included: this says "my
                # mapping is gone", and a producer blocked on it must not be
                # held hostage by how its consumer ended.
                try:
                    write_oncard_release(region, row, wave=wave)
                except Exception:  # noqa: BLE001 -- an unwind must not raise
                    pass
            if bounce is not None:
                if may_free:
                    bounce.close()
                else:
                    # The unwind path of an EXPORTING rank.  Freeing here would
                    # be the same use-after-free on a peer that may still hold
                    # the mapping, and this leg is already refusing the flip --
                    # so the allocation is deliberately LEAKED and said so.
                    # R2-1: a failed exchange is fatal, never retried, so this
                    # cannot accumulate.
                    log(f"{ONCARD_LINE_PREFIX} card={card_uuid} mode={oncard_mode} "
                        f"bounce=leaked bytes={bounce.nbytes} wave={wave} "
                        f"row={row} peer_row={peer_row} -- the exported bounce "
                        f"was NOT freed on this unwind path: the co-located "
                        f"consumer may still have it mapped, and a free under "
                        f"an open IPC mapping is undefined")
            ops.destroy_stream(stream)

    threads.append(threading.Thread(target=guarded(diagonal),
                                    name="weg2-xchg-oncard"))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        # EVERY error is reported, not only the first.  When a pair thread and
        # the diagonal fail together the survivor that gets raised may name the
        # consequence rather than the cause, and the other one would then exist
        # nowhere at all.
        first = errors[0]
        for index, extra in enumerate(errors[1:], start=2):
            note = (f"weg2 xchg leg row={row} wave={wave}: thread {index} of "
                    f"{len(errors)} also failed: {type(extra).__name__}: {extra}")
            log(note)
            if hasattr(first, "add_note"):
                first.add_note(note)
        vote_failure(first)
        raise first

    if not is_source:
        stream = ops.create_stream(device)
        try:
            result.zerofill_bytes = apply_zerofill(ops, stream, descs, rank)
        finally:
            ops.destroy_stream(stream)

    result.pairs = pair_stats
    result.oncard = oncard_stats if on_card else None
    for stats in pair_stats:
        line = stats.line()
        result.lines.append(line)
        log(line)
    if result.oncard is not None:
        line = result.oncard.line()
        result.lines.append(line)
        log(line)
    return result


def _on_card(desc: object) -> bool:
    return int(desc.src_rank) == int(desc.dst_rank)
