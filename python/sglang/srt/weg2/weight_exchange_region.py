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
"""#1273 slice S3 -- the shared exchange region, Gate 0, and the wave gate.

WEG2_REUSE_SPEC_0908 section 6 / S3.  This module owns three things and
nothing else: the ``/dev/shm`` region both ``launch_server`` process trees
map, the **Gate 0** byte-matrix handshake that runs before a byte moves, and
the **wave gate** that closes each wave across all six ranks.  It contains no
CUDA call, imports no torch, and moves no payload -- the transport is S4.

WHY A SEPARATE MODULE FROM ``weight_exchange.py``.  The spec's slice order
(section 6, "Order and parallelism") builds **S1 in parallel with S3** and
says in those words *"no shared file"*, while both slices' file lists name
``weight_exchange.py``.  Splitting the region/gate half into its own module
is what makes that sentence literally true, and it is the only deviation
from S3's file list.  The plan half (S1) imports nothing from here; the two
meet at :func:`gate0_check`, whose ``tag_totals`` / ``tms_tag_bytes``
arguments are the pinned interface (see the TODO there).

TWO IDENTITIES, NOT ONE -- and conflating them is the defect the first cut of
this module shipped (review round 2, F1).  The spec writes ``<epoch>`` in the
region's path, in the semaphore names AND in the gate line's ``epoch=<b.f>``,
but those are not the same token:

* **The region is BOOT-scoped.**  It is created once, before either group
  starts, ``ftruncate``d to 385 MiB and (S4) ``cudaHostRegister``ed once.  Its
  name, its 24 semaphores and its ``registered`` flags are keyed by the **boot
  nonce**, the ``<b>`` half of ``<b.f>``.  That is what closes section 3.8's
  case -- a crashed boot's region can never be adopted by the next boot.
* **The gate, the matrix and the slots are FLIP-scoped.**  Their epoch stamp
  is the full ``credit_epoch`` token ``<boot>.<flip>``, re-stamped on every
  flip by :meth:`XchgRegion.begin_flip`.  Without that re-stamp the previous
  flip's six rows still carry the region's hash at ``gate_seq=3``, and the
  next flip's wave 0 closes ``joined=6/6`` on the first poll with nobody
  having joined anything -- verbatim the stop-loss *"any gate that can pass
  while a rank is absent is not a gate"*.  Nothing on the flip path
  re-creates a file or a semaphore; ``begin_flip`` is two stores.

THREE PROPERTIES THIS FILE EXISTS TO HOLD, each of which is a measured
failure of a predecessor mechanism rather than a precaution:

* **A gate that can pass while a rank is absent is not a gate** (S3's
  stop-loss).  Three independent conditions, all of which must hold for a row
  to count as a join, and all three printed with their denominator: the row's
  SEAL validates (so a half-written row is not a join), the row carries
  **this flip's** epoch hash, and its ``gate_seq`` has reached this wave.  On
  the way out the gate additionally proves the six rows were written by six
  DISTINCT, LIVE pids -- six rows are not six ranks, and a rank that wrote its
  row and then died would otherwise leave a permanent phantom join.
* **Byte counts are agreed before a byte moves** (#802).  ``c26d28172106``
  killed an instance for a corruption that had not happened, because
  ``_dist_exchange`` derived send/recv byte counts independently on each rank
  and never handshook them.  Gate 0 is that handshake: every rank publishes
  BOTH what it will send and what it expects to receive, publishes its own
  rank-local verdict beside them, and every rank checks all 36 cells and all
  six verdicts.  A rank-local disagreement is therefore a GLOBAL refusal --
  a Gate 0 that raised only on the rank that noticed would split the six.
* **A buffer's size never stands for its filled bytes** (#802 rule 2).
  ``bytes_filled`` is written by the producer *after* its own
  ``cudaStreamSynchronize`` and is the only quantity a consumer may copy.
  This module owns the field and its state machine; S4 owns the copy that
  reads it (W70).

THE TWO W-CODES ALLOCATED HERE are W68 and W69 (spec section 7).  Both were
free at the tree of record: the census ``test_weg2_wcode_uniqueness_1263.py``
performs over its own ROOTS returns W50 as the highest assigned code.

RENUMBERED BY THE serve-next5 REPLAY (2026-09-09): W53/W54 here became W68/W69
with the other fourteen, because the serve line had claimed W51-W66 while this
branch was cut.  The W50 maximum quoted above is this branch's OWN base; the
merged maximum is W66 and W84-W82 are what the enumeration returns on it.

LAYOUT (spec S3, with the deviations stated where they occur)::

    /dev/shm/weg2-xchg-<boot nonce>/xchg.bin
      [0        ) header : magic, version, boot hash, n_ranks, slot_bytes,
                           hook_mode, registered bytes, geometry + the 12 slots
      [4 KiB    ) gate   : 6 rows x 64 B   (7 payload words + a seal word)
      [8 KiB    ) matrix : 6 rows x 192 B  (send row, recv row, verdict, seal)
      [64 KiB   ) dir    : the boot-time pointer table + 6 IPC handles (S4)
      [1 MiB    ) data   : 6 cross pairs x 2 slots x 32 MiB = 384 MiB

``SLOT_BYTES = 32 MiB`` is E2's answer (32/64/128 MiB indistinguishable,
< 0.6 %; the smaller slot halves the resident carve-out for free).
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import mmap
import os
import struct
import time
import zlib
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

KIB = 1024
MIB = 1024 * 1024

# --------------------------------------------------------------------------
# Geometry.  Every number here is either a spec constant or derived from one;
# nothing is hand-picked (SLOT_BYTES is E2's measurement, N_RANKS is the two
# groups x three cards of the Weg-2 form).
# --------------------------------------------------------------------------

#: Six ranks: two ``launch_server`` process trees (groups P and D) x three
#: cards.  Rank *n* of either group runs on ``cards[n]`` (spec section 1.3:
#: both groups receive the same ``CUDA_VISIBLE_DEVICES`` uuid string).
N_GROUPS = 2
N_CARDS = 3
N_RANKS = N_GROUPS * N_CARDS

#: The six DIRECTED cross-card pairs.  On-card traffic takes the IPC lane and
#: never a staging slot (spec section 2.2 / S4), so the diagonal is absent.
CROSS_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
)
N_PAIRS = len(CROSS_PAIRS)
SLOTS_PER_PAIR = 2
SLOT_BYTES = 32 * MIB

HEADER_OFF = 0
HEADER_WORDS = 16
HEADER_FIELDS_BYTES = HEADER_WORDS * 8
HEADER_STRUCT = struct.Struct(f"<{HEADER_WORDS}Q")
#: Header word 6, read as SIX INDEPENDENT BYTES: byte *r* is rank *r*'s
#: "I have cudaHostRegister'ed this region" flag.  See
#: :meth:`XchgRegion.mark_registered` for why this is not a bitmap.
REGISTERED_OFF = HEADER_OFF + 6 * 8
#: The boot nonce, as a string, so a region file names its own owner without
#: anyone having to invert a hash.
BOOT_STR_OFF = HEADER_OFF + HEADER_FIELDS_BYTES
BOOT_STR_BYTES = 128
SLOTS_OFF = BOOT_STR_OFF + BOOT_STR_BYTES
SLOT_STRUCT = struct.Struct("<IIQQQQii")  # 48 B, the spec's XchgSlot, in order
SLOT_RECORD_BYTES = SLOT_STRUCT.size
N_SLOTS = N_PAIRS * SLOTS_PER_PAIR

#: A published row is sealed with the top bit set, so a ZEROED word -- an
#: untouched row, or a row whose seal this module deliberately broke before
#: rewriting it -- can never validate as a seal.  A bit position, not a tuned
#: number.
SEAL_MARK = 1 << 63

GATE_OFF = 4 * KIB
#: 64 B per row, because spec section 3.2 says "six 64-byte-aligned shm rows"
#: -- one cache line per rank, so one rank's store never invalidates another's.
GATE_ROW_BYTES = 64
GATE_PAYLOAD_STRUCT = struct.Struct("<7Q")
GATE_SEAL_OFF = GATE_PAYLOAD_STRUCT.size

MATRIX_OFF = 8 * KIB
#: **DEVIATION from the spec's "6 rows x 6 u64", with its reason.** The check
#: the spec mandates is ``send[a][b] == recv[b][a]`` over all 36 cells -- and
#: that is not computable from a send-only matrix, because nothing would carry
#: ``recv``.  Each rank therefore publishes BOTH of its own vectors: what it
#: will send to each peer, and what it expects to receive from each peer.  The
#: row also carries the rank's own GATE-0 VERDICT (see :func:`gate0_check`:
#: a rank-local check that raised only locally would split the six ranks) and
#: a trailing seal word.  Three cache lines, and the row is written and read
#: as a unit.
MATRIX_ROW_BYTES = 192
MATRIX_PAYLOAD_STRUCT = struct.Struct("<23Q")
MATRIX_SEAL_OFF = MATRIX_PAYLOAD_STRUCT.size
#: Word indices inside the matrix payload.
MX_PLAN_HASH = 2 * N_RANKS
MX_EPOCH_HASH = MX_PLAN_HASH + 1
MX_PID = MX_EPOCH_HASH + 1
MX_LOCAL_OK = MX_PID + 1
#: THE ON-CARD MODE, carried in Gate 0 (#1273 S4-fix refusal A, an S6
#: ``must_fix`` carried forward by S5).  S4 closed the SILENT half of that
#: finding -- the slot GEOMETRY, published in the on-card row, refused by name
#: -- and left the LOUD half open: two co-located ranks that disagree about
#: ``ipc`` vs ``host`` fail late and confusingly (a zero handle, or a
#: ``HostBounce(create=False)`` ENOENT) inside the transport, after the flip
#: has begun.  The mode is a per-BOOT arm, so Gate 0 -- which runs before a
#: byte moves and before any ``resume`` -- is where a disagreement costs
#: nothing to refuse.
#: ``0`` means UNSTATED, which is what every rank publishes until S6 passes the
#: launcher's arm down; unstated is not a disagreement, and a gate that refused
#: it would refuse every flip of every boot before S6.
MX_ONCARD_MODE = MX_LOCAL_OK + 1

#: The mode values, as one word.  Not a string in a row: the row is fixed-width
#: u64s and a hash of "ipc" would be a second identity for a two-valued field.
ONCARD_MODE_UNSTATED = 0
ONCARD_MODE_WORD = {"ipc": 1, "host": 2}
ONCARD_MODE_NAMES = {0: "unstated", 1: "ipc", 2: "host"}

DIR_OFF = 64 * KIB
DATA_OFF = 1 * MIB
DATA_BYTES = N_PAIRS * SLOTS_PER_PAIR * SLOT_BYTES
REGION_BYTES = DATA_OFF + DATA_BYTES

XCHG_MAGIC = int.from_bytes(b"WEG2XCHG", "little")
XCHG_VERSION = 2

#: Slot states (spec S3's ``XchgSlot.state``).
SLOT_FREE = 0
SLOT_FILLING = 1
SLOT_PRODUCED = 2
SLOT_DRAINING = 3
SLOT_STATE_NAMES = {
    SLOT_FREE: "FREE",
    SLOT_FILLING: "FILLING",
    SLOT_PRODUCED: "PRODUCED",
    SLOT_DRAINING: "DRAINING",
}

#: Gate row states.  ``GATE_IDLE`` is the zeroed region: a rank that has never
#: written a row is distinguishable from one that voted.
GATE_IDLE = 0
GATE_JOINED = 1
GATE_FAILED = 2

#: Spec section 3.2: "Cost: 3 gates x 0.5 ms poll granularity."
GATE_POLL_S = 0.0005

SHM_ROOT = "/dev/shm"
#: The one name family this mechanism creates.  ``launcher.SHM_OWN_PREFIXES``
#: carries it (and ``sem.`` + it, for the named semaphores, which glibc
#: materialises as ``/dev/shm/sem.<name>``) so the #1233 fix-8 residue sweep
#: can see a crashed boot's region instead of leaving it resident.
REGION_PREFIX = "weg2-xchg-"
REGION_FILE = "xchg.bin"
ENV_REGION_PATH = "SGLANG_WEG2_XCHG_REGION"
#: The BOOT NONCE, not a flip epoch: the region this names outlives every flip
#: of the boot.  The flip token travels on the RPC (spec section 3.1).
ENV_REGION_BOOT = "SGLANG_WEG2_XCHG_BOOT"

HOOK_MODE_UNKNOWN = 0
HOOK_MODE_PRELOAD = 1
HOOK_MODE_TORCH = 2
HOOK_MODE_NAMES = {
    HOOK_MODE_UNKNOWN: "unknown",
    HOOK_MODE_PRELOAD: "preload",
    HOOK_MODE_TORCH: "torch",
}


# --------------------------------------------------------------------------
# Refusals.  The NAME identifies the refusal, the number is a label humans and
# greps read (spec section 7).
# --------------------------------------------------------------------------


class Weg2XchgPlanDisagree(RuntimeError):
    """W68: two things that must agree about this exchange do not.

    Raised on EVERY reader, never only on the rank that happens to notice --
    #802's rule is that the disagreement is a property of the pair, so both
    halves must stop.  The message names the cell, both values and their
    difference, because "plan mismatch" without the cell is a postmortem
    nobody can start from.

    **DEVIATION, stated: this code is used more widely than spec section 7's
    row for it.**  Section 7 assigns W68 to Gate 0 (asymmetric matrix, a
    per-tag total != ``tms_tag_bytes``, a plan hash != the front's).  It is
    also raised here for every OTHER disagreement about the shared object
    itself: a file that is not an exchange region, a region built by a
    different version of this module, a region whose header geometry is not
    this module's, a region belonging to another boot, a flip token that is
    not of this boot, a publish onto a slot this flip never claimed, and a
    ``bytes_filled`` beyond the slot's capacity.  The alternative would be
    four more codes for one sentence each; the reason they are one code is
    that every one of them means *the two sides do not agree about what this
    region is*, which is the same refusal one layer down.  W70
    ``Weg2XchgShortPiece`` is NOT taken here: it is the CONSUMER's per-piece
    check and belongs to S4.
    """


class Weg2XchgGateTimeout(RuntimeError):
    """W69: a rank did not join a gate, or joined it and then died.

    Names every non-joiner by group, rank, row, pid, the wave it was last
    seen at, the wave it was needed at, and whether its pid is still in
    ``/proc`` -- the poll knows all of that, which is precisely why spec
    section 3.2 refuses a gloo ``monitored_barrier`` for this gate.  It is
    also the refusal when a row that DID join was written by a pid that is no
    longer alive, or when the six rows are not six distinct pids: six rows are
    not six ranks.
    """


# --------------------------------------------------------------------------
# Identity helpers.
# --------------------------------------------------------------------------


def epoch_hash(token: str) -> int:
    """A stable 64-bit hash of a boot nonce or a flip epoch token.

    ``hash()`` is NOT usable: PYTHONHASHSEED randomises it per process, and
    the six ranks are six processes -- they would each stamp a different
    number into shared memory and every epoch filter in this file would
    reject every peer.  blake2b is stable across processes and boots.

    Never zero: zero is the zeroed region, i.e. "nobody wrote here", and a
    token that collided with it would make an untouched row indistinguishable
    from a joined one.
    """
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value or 1


def split_flip_epoch(flip_epoch: str) -> Tuple[str, int]:
    """``"<boot nonce>.<flip index>"`` -> ``(boot nonce, flip index)``.

    The flip index must parse as an integer, because
    :meth:`XchgRegion.begin_flip` refuses a flip that does not advance -- a
    rank re-entering a flip it has already run is a bug, and it is the one
    ordering error the region can catch rank-locally.
    """
    token = str(flip_epoch)
    boot, _, flip = token.rpartition(".")
    if not boot or not flip:
        raise ValueError(
            f"flip epoch {token!r} is not '<boot nonce>.<flip index>' -- "
            f"credit_epoch composes the two and this module splits them"
        )
    try:
        index = int(flip)
    except ValueError:
        raise ValueError(
            f"flip epoch {token!r} has a non-integer flip index {flip!r}"
        ) from None
    return boot, index


def rank_row(group: str, rank: int) -> int:
    """Row index of one rank in the six-row gate and matrix areas.

    ``group`` is ``"P"`` or ``"D"``; the row index is the identity, and the
    group/rank pair is recoverable from it (:func:`row_group_rank`), so no row
    has to carry them as fields.
    """
    g = str(group).upper()
    if g not in ("P", "D"):
        raise ValueError(f"group must be 'P' or 'D', not {group!r}")
    if not 0 <= int(rank) < N_CARDS:
        raise ValueError(f"rank must be 0..{N_CARDS - 1}, not {rank!r}")
    return (0 if g == "P" else 1) * N_CARDS + int(rank)


def row_group_rank(row: int) -> Tuple[str, int]:
    """Inverse of :func:`rank_row`."""
    if not 0 <= int(row) < N_RANKS:
        raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
    return ("P" if int(row) < N_CARDS else "D", int(row) % N_CARDS)


def pair_id(src_card: int, dst_card: int) -> int:
    """Index of the directed cross-card pair ``src_card -> dst_card``."""
    try:
        return CROSS_PAIRS.index((int(src_card), int(dst_card)))
    except ValueError:
        raise ValueError(
            f"({src_card} -> {dst_card}) is not a cross-card pair; the on-card "
            f"lane takes the IPC bounce buffer and never a staging slot"
        ) from None


def region_dir(boot_nonce: str, shm_root: str = SHM_ROOT) -> str:
    return os.path.join(shm_root, f"{REGION_PREFIX}{boot_nonce}")


def region_path(boot_nonce: str, shm_root: str = SHM_ROOT) -> str:
    return os.path.join(region_dir(boot_nonce, shm_root), REGION_FILE)


def fence_budget_s() -> float:
    """The gate's deadline.  ONE budget for the whole Weg-2 flip, not a second.

    Spec section 3.2: *"No second timeout constant."*
    ``WEG2_GROUP_FENCE_BUDGET_S`` (``weight_updater.py:76``) already governs
    the group fence and the credit wait; the cross-group gate is the third
    consumer of the same number, never of a literal of its own.  Imported
    lazily so this module stays torch-free and importable under
    ``CUDA_VISIBLE_DEVICES=""``.
    """
    from sglang.srt.managers.scheduler_components.weight_updater import (
        WEG2_GROUP_FENCE_BUDGET_S,
    )

    return float(WEG2_GROUP_FENCE_BUDGET_S)


def _seal(payload: bytes) -> int:
    """The seal word of a published row.

    WHY A SEAL EXISTS AT ALL.  A gate row and a matrix row are the two objects
    in this region written by one rank and read by five WITHOUT a semaphore
    between them (slots have one; these deliberately do not, because a poll
    that cannot deadlock is the whole reason section 3.2 refuses a process
    group here).  ``struct.pack_into`` writes fields in ascending order, so a
    reader can otherwise observe half a row: the previous flip's ``ok`` beside
    this flip's ``gate_seq``, which reads as a joined-and-failed rank and
    aborts a healthy flip.  Correctness must not rest on which field the
    packer happens to store last.

    So: the writer zeroes the seal, writes the payload, then writes the seal;
    the reader takes the row in one slice and counts it only if the seal
    matches.  ``crc32`` is the cheapest thing that detects a torn row (this is
    a tear detector, not a security property), and :data:`SEAL_MARK` keeps a
    zeroed word from ever validating.
    """
    return zlib.crc32(payload) | SEAL_MARK


def _pid_alive(pid: int, proc_root: str) -> bool:
    return bool(pid) and os.path.isdir(os.path.join(proc_root, str(int(pid))))


# --------------------------------------------------------------------------
# Records.
# --------------------------------------------------------------------------


class XchgSlot:
    """One staging slot, exactly the spec's ``struct XchgSlot``."""

    __slots__ = (
        "state", "pair_id", "epoch_hash", "seq", "bytes_filled",
        "checksum", "producer_pid", "consumer_pid", "index",
    )

    def __init__(self, state, pair, eh, seq, bytes_filled, checksum, ppid, cpid, index=-1):
        self.state = int(state)
        self.pair_id = int(pair)
        self.epoch_hash = int(eh)
        self.seq = int(seq)
        self.bytes_filled = int(bytes_filled)
        self.checksum = int(checksum)
        self.producer_pid = int(ppid)
        self.consumer_pid = int(cpid)
        self.index = int(index)

    @property
    def state_name(self) -> str:
        return SLOT_STATE_NAMES.get(self.state, f"?{self.state}")

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"XchgSlot(index={self.index} pair={self.pair_id} state={self.state_name} "
            f"seq={self.seq} bytes_filled={self.bytes_filled} "
            f"producer_pid={self.producer_pid} consumer_pid={self.consumer_pid})"
        )


class GateRow:
    """One rank's row of the wave gate."""

    __slots__ = ("row", "gate_seq", "ok", "pid", "ts_ns", "state", "epoch_hash", "sealed")

    def __init__(self, row, gate_seq, ok, pid, ts_ns, state, eh, sealed=True):
        self.row = int(row)
        self.gate_seq = int(gate_seq)
        self.ok = bool(ok)
        self.pid = int(pid)
        self.ts_ns = int(ts_ns)
        self.state = int(state)
        self.epoch_hash = int(eh)
        self.sealed = bool(sealed)

    @property
    def group_rank(self) -> Tuple[str, int]:
        return row_group_rank(self.row)

    def wave_seen(self) -> int:
        """The wave this row last joined, or -1 for a row nobody ever wrote."""
        return self.gate_seq - 1


class MatrixRow:
    """One rank's published send and recv vectors, verdict and plan hash."""

    __slots__ = ("row", "send", "recv", "plan_hash", "epoch_hash", "pid",
                 "local_ok", "sealed", "oncard_mode")

    def __init__(self, row, send, recv, plan_hash, eh, pid, local_ok=True,
                 sealed=True, oncard_mode=ONCARD_MODE_UNSTATED):
        self.row = int(row)
        self.send = tuple(int(x) for x in send)
        self.recv = tuple(int(x) for x in recv)
        self.plan_hash = int(plan_hash)
        self.epoch_hash = int(eh)
        self.pid = int(pid)
        self.local_ok = bool(local_ok)
        self.sealed = bool(sealed)
        self.oncard_mode = int(oncard_mode)

    @property
    def oncard_mode_name(self) -> str:
        return ONCARD_MODE_NAMES.get(self.oncard_mode, f"unknown({self.oncard_mode})")


# --------------------------------------------------------------------------
# The region.
# --------------------------------------------------------------------------


class XchgRegion:
    """The mapped exchange region.  Every rank of both groups maps this file.

    The object is a VIEW, never a cache: every read goes to the mapping, so a
    peer's store is visible on the next read.  The gate and matrix rows carry
    their own seal (:func:`_seal`), which is what makes a row's publication
    complete before a reader can count it; the slots' publication is instead
    ordered by the ``empty``/``full`` semaphore pair S4 owns, which is
    memory-synchronising.

    The view is BOOT-scoped in its file and semaphores and FLIP-scoped in its
    stamps: :meth:`begin_flip` is the only thing that makes gate rows, matrix
    rows and slots writable, and every one of them refuses while no flip is
    bound.
    """

    def __init__(self, path: str, mm: mmap.mmap, fd: int, boot_nonce: str):
        self.path = path
        self._mm = mm
        self._fd = fd
        self.boot_nonce = str(boot_nonce)
        self.boot_hash = epoch_hash(self.boot_nonce)
        #: The flip token ``<boot>.<flip>``, or "" while no flip is bound.
        self.epoch = ""
        #: The flip token's hash, or 0 while no flip is bound.  Every joined
        #: count in this module filters on it.
        self.epoch_hash = 0
        self.flip_index = -1
        #: This rank's own row and current wave, for the failure vote.  See
        #: :meth:`bind`.
        self.row: Optional[int] = None
        self.wave: int = 0
        #: Cached mapping base address; see :meth:`base_address`.
        self._base: int = 0

    # ---- lifecycle ------------------------------------------------------

    @classmethod
    def create(
        cls,
        boot_nonce: str,
        *,
        path: str = "",
        shm_root: str = SHM_ROOT,
        hook_mode: int = HOOK_MODE_UNKNOWN,
    ) -> "XchgRegion":
        """Create the region for ONE BOOT and zero its gates.

        ``O_EXCL``: re-creating a region that already exists would re-zero
        every gate row, matrix row and slot record of a LIVE boot and report
        success.  The launcher's residue sweep is what removes a dead boot's
        region, and it refuses the boot outright while a holder lives; if a
        region is still here at create time, that ordering was broken and this
        refuses by name rather than papering over it.

        ``ftruncate`` to the full :data:`REGION_BYTES`.  On tmpfs the file is
        sparse at this point, but it does NOT stay sparse: S4
        ``cudaHostRegister``s the whole region, and pinning populates every
        page, so the +0.38 GiB of section 0.2's ledger term lands in full --
        and PINNED, i.e. not reclaimable -- at registration time, not
        incrementally as the transport writes.  The magnitude is the ledger's;
        the timing is stated here because the ledger reader will look here.
        """
        target = path or region_path(boot_nonce, shm_root)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree create path={target}: a region for boot "
                f"{boot_nonce!r} already exists.  Creating it again would re-zero a "
                f"live boot's gate rows, matrix rows and slots and report success -- "
                f"the launcher's shm residue sweep owns removing a dead boot's region, "
                f"and it refuses the boot while a holder lives"
            ) from None
        try:
            os.ftruncate(fd, REGION_BYTES)
            mm = mmap.mmap(fd, REGION_BYTES, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(fd)
            raise
        region = cls(target, mm, fd, str(boot_nonce))
        # Zero everything below the payload. The DATA area is deliberately NOT
        # zeroed (385 MiB of pointless page faults); a slot's content is only
        # ever read through its own header record, whose flip epoch hash is
        # checked (see `claim_produced`).
        mm[0:DATA_OFF] = b"\x00" * DATA_OFF
        header = [
            XCHG_MAGIC, XCHG_VERSION, region.boot_hash, N_RANKS,
            SLOT_BYTES, int(hook_mode), 0, REGION_BYTES,
            N_PAIRS, SLOTS_PER_PAIR, time.time_ns(), os.getpid(),
            GATE_ROW_BYTES, MATRIX_ROW_BYTES, DATA_OFF, 0,
        ]
        HEADER_STRUCT.pack_into(mm, HEADER_OFF, *header)
        raw = str(boot_nonce).encode("utf-8")[: BOOT_STR_BYTES - 1]
        mm[BOOT_STR_OFF: BOOT_STR_OFF + BOOT_STR_BYTES] = raw.ljust(BOOT_STR_BYTES, b"\x00")
        for index in range(N_SLOTS):
            region._write_slot_record(index, XchgSlot(
                SLOT_FREE, index // SLOTS_PER_PAIR, 0, 0, 0, 0, 0, 0, index))
        return region

    @classmethod
    def open(cls, path: str, *, expect_boot: str) -> "XchgRegion":
        """Map an existing region.  Refuses a foreign or stale file BY NAME.

        ``expect_boot`` is REQUIRED, not defaulted.  A default would make the
        anti-adoption guard opt-in, and a guard every future caller has to
        remember is the guard that boot weg2rg2's three terminal credit
        counters got wrong: the point of the epoch is that nobody has to
        remember it.
        """
        fd = os.open(path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, REGION_BYTES, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(fd)
            raise

        def refuse(why: str) -> "Weg2XchgPlanDisagree":
            mm.close()
            os.close(fd)
            return Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree path={path}: {why}"
            )

        f = HEADER_STRUCT.unpack_from(mm, HEADER_OFF)
        if f[0] != XCHG_MAGIC:
            raise refuse(f"magic {f[0]:#x} is not {XCHG_MAGIC:#x} -- not an exchange region")
        if f[1] != XCHG_VERSION:
            raise refuse(
                f"region version {f[1]} != {XCHG_VERSION} -- two builds of the "
                f"exchange in one boot"
            )
        # The geometry is the byte identity of the staging layout. Two builds
        # that share a version but not a layout would each address the slots
        # with their OWN constants while printing the file's -- so the header's
        # geometry is CHECKED here, and every offset in this module is then
        # computed from the module constants alone (one source of truth).
        for name, got, want in (
            ("n_ranks", f[3], N_RANKS),
            ("slot_bytes", f[4], SLOT_BYTES),
            ("region_bytes", f[7], REGION_BYTES),
            ("n_pairs", f[8], N_PAIRS),
            ("slots_per_pair", f[9], SLOTS_PER_PAIR),
            ("gate_row_bytes", f[12], GATE_ROW_BYTES),
            ("matrix_row_bytes", f[13], MATRIX_ROW_BYTES),
            ("data_off", f[14], DATA_OFF),
        ):
            if int(got) != int(want):
                raise refuse(
                    f"header {name}={got} != this build's {want} -- the region's "
                    f"geometry is not the geometry this process would address it with"
                )
        raw = bytes(mm[BOOT_STR_OFF: BOOT_STR_OFF + BOOT_STR_BYTES])
        boot = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
        if epoch_hash(boot) != f[2]:
            raise refuse(
                f"the region names boot {boot!r} whose hash is {epoch_hash(boot):#x}, "
                f"but its header carries {f[2]:#x} -- the string and the hash the "
                f"creator wrote disagree, so neither can be trusted as the identity"
            )
        if boot != str(expect_boot):
            raise refuse(
                f"region carries boot nonce {boot!r}, this rank was launched for "
                f"{str(expect_boot)!r} -- a previous boot's region, never adopted "
                f"as this one's"
            )
        return cls(path, mm, fd, boot)

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

    def __enter__(self) -> "XchgRegion":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- the flip stamp -------------------------------------------------

    def begin_flip(self, flip_epoch: str) -> int:
        """Stamp this VIEW with the flip token ``<boot>.<flip>``.  Two stores.

        THE REGION IS BOOT-SCOPED AND THE STAMP IS FLIP-SCOPED, and this method
        is the seam.  Nothing here creates a file, truncates, registers or
        opens a semaphore -- a per-flip region would put a 385 MiB
        ``ftruncate``, a ``cudaHostRegister`` and 24 ``sem_open`` calls on a
        1.5 s transport's critical path, and there is no term for that in
        section 0.3's budget.

        Skipping this re-stamp is the defect it exists to prevent: the
        previous flip leaves all six gate rows at ``gate_seq=3`` carrying the
        region's hash, so the next flip's wave 0 counts ``joined=6/6`` on its
        first poll and moves bytes with nobody having joined.  Every joined
        count in this module filters on the value this method writes.

        Refuses a flip token of ANOTHER boot (the region could not be its
        region) and a flip index that does not advance (a rank re-entering a
        flip it has already run).
        """
        boot, index = split_flip_epoch(flip_epoch)
        if boot != self.boot_nonce:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree begin_flip epoch={flip_epoch!r}: its boot "
                f"nonce {boot!r} is not this region's {self.boot_nonce!r} -- a flip of "
                f"another boot can never be run on this region"
            )
        if index <= self.flip_index:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree begin_flip epoch={flip_epoch!r}: flip index "
                f"{index} does not advance past {self.flip_index}, which this view has "
                f"already run -- a flip is entered once, and re-entering an old one "
                f"would re-adopt its rows as this flip's joins"
            )
        self.epoch = str(flip_epoch)
        self.epoch_hash = epoch_hash(flip_epoch)
        self.flip_index = index
        self.wave = 0
        return self.epoch_hash

    def bind(self, row: int, wave: Optional[int] = None) -> None:
        """Declare which row this process owns, so a refusal can VOTE.

        Spec section 3.6: *"a rank that cannot derive its plan votes
        ``ok=False`` rather than skipping descriptors."*  A refusal raised
        inside this module -- a bad publish, a dead producer, a Gate 0
        disagreement -- must reach the other five as a vote, not as silence:
        silence costs them the whole 120 s fence budget before they learn
        anything, which is exactly the fast named propagation section 3.2
        sells against ``monitored_barrier``.

        S4/S6 call this once with the rank's row, and again with each wave as
        the transport enters it.  While it is unbound, every refusal in this
        module SAYS SO in its message rather than leaving the gap silent.
        """
        if not 0 <= int(row) < N_RANKS:
            raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
        self.row = int(row)
        if wave is not None:
            self.wave = int(wave)

    def vote_failure_note(self) -> str:
        """Publish ``ok=False`` for this rank's current wave; describe what happened.

        Returns the sentence appended to every refusal this module raises, so
        the log says whether the peers were told or are waiting.
        """
        if self.row is None or not self.epoch_hash:
            return (
                "[NO failure vote published: this view is not bound to a row and a flip "
                "(XchgRegion.bind / begin_flip), so the five peers learn nothing until "
                "their gate reaches the fence budget]"
            )
        try:
            self.write_gate_row(self.row, self.wave, False)
        except Exception as exc:  # pragma: no cover - the mapping is gone
            return f"[failure vote FAILED to publish: {exc!r}]"
        return (
            f"[failure voted: row={self.row} wave={self.wave} ok=False -- the peers' "
            f"wave gate refuses now, not at the fence budget]"
        )

    def _refuse(self, cls, message: str):
        """Vote false if bound, then build the refusal with that note attached."""
        return cls(f"{message} {self.vote_failure_note()}")

    def _require_flip(self, what: str) -> None:
        if not self.epoch_hash:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree {what}: no flip is bound to this view. "
                f"XchgRegion.begin_flip('<boot>.<flip>') stamps the gate rows, matrix "
                f"rows and slots; writing them with the boot's own identity would let "
                f"the previous flip's rows count as this flip's joins"
            )

    # ---- header ---------------------------------------------------------

    def header(self) -> Dict[str, int]:
        f = HEADER_STRUCT.unpack_from(self._mm, HEADER_OFF)
        return {
            "magic": f[0], "version": f[1], "boot_hash": f[2], "n_ranks": f[3],
            "slot_bytes": f[4], "hook_mode": f[5],
            # NOT a bitmap: six independent bytes, one per rank. The name says
            # so, because a caller that popcount'ed a bitmap here would get the
            # right answer only by the coincidence that 0x01 has one bit set.
            "registered_bytes": f[6],
            "region_bytes": f[7], "n_pairs": f[8], "slots_per_pair": f[9],
            "created_ns": f[10], "creator_pid": f[11],
            "gate_row_bytes": f[12], "matrix_row_bytes": f[13],
            "data_off": f[14], "n_sems": f[15],
        }

    def set_sem_count(self, n: int) -> None:
        """Record how many named semaphores were created for this region."""
        struct.pack_into("<Q", self._mm, HEADER_OFF + 15 * 8, int(n))

    # ---- addresses and the dir area: the S4 seam ------------------------
    #
    # These three exist ONLY so the transport (S4) does not open a SECOND
    # mapping of this file.  Two mappings of one region is second bookkeeping
    # beside a truth this object already owns (upstream-minimal law): the two
    # would have different base addresses, and ``cudaHostRegister`` on one of
    # them would leave the other's pages unpinned while every count here said
    # "registered".  So the transport asks this object for the address it
    # already has, and writes into the dir area through a BOUNDED view rather
    # than through ``_mm``.

    def base_address(self) -> int:
        """The mapping's base address -- what ``cudaHostRegister`` pins.

        The whole region is registered once per rank, not slot by slot: the
        header, gate, matrix and dir areas are read and written by the CPU on
        the flip path, and a partially pinned mapping would make which half is
        pinned an accident of the slot arithmetic.

        Computed ONCE and cached, and the temporary that computes it is dropped
        inside this method.  ``from_buffer`` EXPORTS the mmap's buffer for as
        long as the ctypes object lives, and an exported buffer makes
        ``mmap.close()`` raise ``BufferError``; a cached integer keeps that
        window to the width of one statement instead of leaving it open for
        every caller that ever asked for an address.
        """
        if self._base:
            return self._base
        holder = ctypes.c_char.from_buffer(self._mm)
        self._base = ctypes.addressof(holder)
        del holder
        return self._base

    def slot_address(self, pair: int, slot: int) -> int:
        """Absolute address of one staging slot's payload."""
        return self.base_address() + self.data_offset(pair, slot)

    def dir_view(self):
        """The ``[64 KiB, 1 MiB)`` dir area, as a writable ``c_ubyte`` array.

        BOUNDED BY CONSTRUCTION: the array is exactly ``DATA_OFF - DIR_OFF``
        long, so an arithmetic error inside the transport's own sub-layout
        raises ``IndexError`` instead of reaching the header, the gate rows,
        the matrix rows or a payload slot.  The sub-layout of this area (the S6
        pointer table, the six IPC handles and the on-card handshake rows) is
        owned by the transport, which is why this hands back storage and not a
        set of typed accessors.

        A ``ctypes`` array over the address rather than a ``memoryview``, and
        that is not a style choice: a memoryview EXPORTS the mmap's buffer, and
        an exported buffer makes ``mmap.close()`` raise -- so every caller
        would have to release its view on every path, including the raising
        ones, and one missed release turns :meth:`close` into a
        ``BufferError`` at teardown.  ``from_address`` exports nothing.
        """
        return (ctypes.c_ubyte * (DATA_OFF - DIR_OFF)).from_address(
            self.base_address() + DIR_OFF)

    def dir_address(self) -> int:
        """Absolute address of the dir area -- for a CUDA call that needs a
        pointer rather than a Python buffer."""
        return self.base_address() + DIR_OFF

    def mark_registered(self, row: int, *, log: Optional[Callable[[str], None]] = None) -> int:
        """Record that one rank has ``cudaHostRegister``ed this region (S4).

        ONE BYTE PER RANK, at that rank's own address -- not a bitmap, and the
        difference is a MEASURED defect rather than a style preference.  The
        first version of this method was ``bits |= 1 << row`` on a shared u64:
        a read-modify-write, which pure Python on an mmap cannot make atomic.
        The six-rank hermetic double reported ``registered=5/6`` on the remote
        desk (2026-09-08) because two ranks read the same word and the later
        store dropped the earlier rank's bit.  An under-count here reads as
        "one rank never registered", i.e. it turns an armed region into an
        apparently unarmed one -- and the same class of lost update could just
        as easily hide a rank that genuinely did not register.

        A per-rank byte needs no atomic: distinct addresses, one writer each,
        which is the same law the gate rows and matrix rows already obey.
        Idempotent by construction, so a rank that registers twice cannot make
        the denominator read 7/6.

        ``log`` is how S3's acceptance line can ever be TRUE.  The line the
        launcher prints at create time necessarily says ``registered=0/6``
        (no rank has registered yet), so if nothing re-emitted it,
        ``registered=6/6`` -- the spec's acceptance -- would be unobservable.
        The rank whose byte completes the six re-emits it.
        """
        if not 0 <= int(row) < N_RANKS:
            raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
        self._mm[REGISTERED_OFF + int(row)] = 1
        count = self.registered_count()
        if log is not None and count == N_RANKS:
            log(region_line(self))
        return count

    def registered_rows(self) -> List[int]:
        """Which ranks have registered -- namable, not merely a total."""
        return [r for r in range(N_RANKS) if self._mm[REGISTERED_OFF + r]]

    def registered_count(self) -> int:
        return len(self.registered_rows())

    # ---- slots ----------------------------------------------------------

    def _slot_off(self, index: int) -> int:
        if not 0 <= int(index) < N_SLOTS:
            raise ValueError(f"slot index must be 0..{N_SLOTS - 1}, not {index!r}")
        return SLOTS_OFF + int(index) * SLOT_RECORD_BYTES

    @staticmethod
    def slot_index(pair: int, slot: int) -> int:
        if not 0 <= int(pair) < N_PAIRS:
            raise ValueError(f"pair must be 0..{N_PAIRS - 1}, not {pair!r}")
        if not 0 <= int(slot) < SLOTS_PER_PAIR:
            raise ValueError(f"slot must be 0..{SLOTS_PER_PAIR - 1}, not {slot!r}")
        return int(pair) * SLOTS_PER_PAIR + int(slot)

    def data_offset(self, pair: int, slot: int) -> int:
        """Byte offset of one slot's payload inside the region."""
        return DATA_OFF + self.slot_index(pair, slot) * SLOT_BYTES

    def read_slot(self, pair: int, slot: int) -> XchgSlot:
        index = self.slot_index(pair, slot)
        fields = SLOT_STRUCT.unpack_from(self._mm, self._slot_off(index))
        return XchgSlot(*fields, index=index)

    def _write_slot_record(self, index: int, rec: XchgSlot) -> None:
        """Rewrite one 48-byte slot record.  SINGLE WRITER AT A TIME.

        Producer and consumer both write this record, and it is not atomic --
        but they never write it concurrently, because the ``empty``/``full``
        semaphore pair (S4) makes them strictly alternate: the producer owns
        the slot from ``sem_wait(empty)`` to ``sem_post(full)``, the consumer
        from ``sem_wait(full)`` to ``sem_post(empty)``.  ``sem_post`` /
        ``sem_wait`` are memory-synchronising, which is why this record needs
        no seal word while the gate and matrix rows -- read by five peers with
        no semaphore between them -- do (:func:`_seal`).  The struct is the
        spec's 48-byte ``XchgSlot`` verbatim and is not widened here.

        That is the same reason the gate and matrix areas give every rank its
        own row: a shared word with two writers and no atomic is a lost
        update, which this module already measured once (see
        :meth:`mark_registered`).
        """
        SLOT_STRUCT.pack_into(
            self._mm, self._slot_off(index),
            rec.state, rec.pair_id, rec.epoch_hash, rec.seq,
            rec.bytes_filled, rec.checksum, rec.producer_pid, rec.consumer_pid,
        )

    def begin_fill(self, pair: int, slot: int, seq: int, *, producer_pid: int = 0) -> XchgSlot:
        """Claim a slot for this flip and mark it FILLING.  No bytes yet."""
        self._require_flip(f"begin_fill pair={pair} slot={slot}")
        index = self.slot_index(pair, slot)
        rec = XchgSlot(SLOT_FILLING, pair, self.epoch_hash, int(seq), 0, 0,
                       int(producer_pid or os.getpid()), 0, index)
        self._write_slot_record(index, rec)
        return rec

    def publish(self, pair: int, slot: int, bytes_filled: int, *, checksum: int = 0) -> XchgSlot:
        """Publish a filled slot.  **Call only after the producer's sync.**

        ``bytes_filled`` is the #802 handshake (rule 2): the consumer copies
        this number, never the slot's capacity.  Writing it here, after the
        caller's ``cudaStreamSynchronize``, is what makes the number mean
        "these bytes have landed" rather than "these bytes were issued".

        UNWIND CONTRACT (section 3.4 invariant b, lock order PCIe key ->
        semaphore -> shm gate): a refusal raised here happens with the slot's
        ``full`` semaphore UNPOSTED and the slot still FILLING, so the
        consumer would block on it.  That is why the refusal votes ``ok=False``
        into this rank's gate row on the way out -- the consumer's wave gate
        then refuses immediately instead of at the fence budget.  The caller
        must not swallow this and must not post ``full``.
        """
        self._require_flip(f"publish pair={pair} slot={slot}")
        index = self.slot_index(pair, slot)
        rec = self.read_slot(pair, slot)
        if rec.state != SLOT_FILLING or rec.epoch_hash != self.epoch_hash:
            raise self._refuse(
                Weg2XchgPlanDisagree,
                f"W68 Weg2XchgPlanDisagree publish pair={pair} slot={slot}: the slot is "
                f"{rec.state_name} at epoch_hash={rec.epoch_hash:#x}, expected FILLING at "
                f"{self.epoch_hash:#x} -- a publish onto a slot this flip never claimed",
            )
        if int(bytes_filled) > SLOT_BYTES:
            raise self._refuse(
                Weg2XchgPlanDisagree,
                f"W68 Weg2XchgPlanDisagree publish pair={pair} slot={slot}: "
                f"bytes_filled={bytes_filled} exceeds slot_bytes={SLOT_BYTES}",
            )
        rec.bytes_filled = int(bytes_filled)
        rec.checksum = int(checksum)
        rec.state = SLOT_PRODUCED
        self._write_slot_record(index, rec)
        return rec

    def claim_produced(
        self,
        pair: int,
        slot: int,
        *,
        consumer_pid: int = 0,
        proc_root: str = "/proc",
    ) -> Optional[XchgSlot]:
        """Take a PRODUCED slot for draining, or refuse/skip it by name.

        Three outcomes, and the two that are not "here are your bytes" are the
        reason this method exists rather than a bare read:

        * **PRODUCED at this flip's epoch** -> returned, state DRAINING, and it
          does NOT matter whether the producer is still alive.  A producer that
          published and then died has already delivered; killing the consumer
          over it would destroy bytes that are sitting in shared memory
          (judges' graft 5).
        * **any state at a FOREIGN epoch** -> the record is zeroed and ``None``
          is returned.  This is ``host_ring.cpp:285-291``'s rule ported to
          slots: a previous boot's or a previous flip's leftovers are never
          read as this flip's funding.
        * **FILLING at this epoch whose producer is gone from /proc** -> W69.
          Nobody will ever post this slot, and a bounded wait that ends in a
          timeout would name the wait rather than the death.
        """
        self._require_flip(f"claim_produced pair={pair} slot={slot}")
        rec = self.read_slot(pair, slot)
        if rec.epoch_hash != self.epoch_hash:
            if rec.state != SLOT_FREE or rec.bytes_filled:
                self._write_slot_record(rec.index, XchgSlot(
                    SLOT_FREE, pair, 0, 0, 0, 0, 0, 0, rec.index))
            return None
        if rec.state == SLOT_FILLING:
            if not _pid_alive(rec.producer_pid, proc_root):
                raise self._refuse(
                    Weg2XchgGateTimeout,
                    f"W69 Weg2XchgGateTimeout slot pair={pair} slot={slot} state=FILLING "
                    f"epoch={self.epoch} producer_pid={rec.producer_pid} "
                    f"alive_in_proc=no seq={rec.seq} bytes_filled={rec.bytes_filled} -- the "
                    f"producer died mid-fill; these bytes will never be posted",
                )
            return None
        if rec.state != SLOT_PRODUCED:
            return None
        rec.state = SLOT_DRAINING
        rec.consumer_pid = int(consumer_pid or os.getpid())
        self._write_slot_record(rec.index, rec)
        return rec

    def release_slot(self, pair: int, slot: int) -> None:
        """Return a drained slot to FREE, keeping this flip's epoch stamp."""
        self._require_flip(f"release_slot pair={pair} slot={slot}")
        index = self.slot_index(pair, slot)
        self._write_slot_record(index, XchgSlot(
            SLOT_FREE, pair, self.epoch_hash, 0, 0, 0, 0, 0, index))

    # ---- gate rows ------------------------------------------------------

    def _gate_off(self, row: int) -> int:
        if not 0 <= int(row) < N_RANKS:
            raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
        return GATE_OFF + int(row) * GATE_ROW_BYTES

    def read_gate_row(self, row: int) -> GateRow:
        off = self._gate_off(row)
        raw = bytes(self._mm[off: off + GATE_ROW_BYTES])
        payload = raw[:GATE_SEAL_OFF]
        (seal,) = struct.unpack_from("<Q", raw, GATE_SEAL_OFF)
        f = GATE_PAYLOAD_STRUCT.unpack(payload)
        return GateRow(row, f[0], bool(f[1]), f[2], f[3], f[4], f[5],
                       sealed=(seal == _seal(payload)))

    def write_gate_row(self, row: int, wave: int, ok: bool, *, pid: int = 0,
                       ts_ns: int = 0) -> GateRow:
        """Publish this rank's vote for ``wave``.  Seal broken, then re-made.

        The row carries THIS FLIP's epoch hash, which is what makes a row left
        behind by the previous flip uncountable: it has the previous flip's
        hash, and every joined-count in this module filters on the hash.

        Three stores in this order and no other: break the old seal, write the
        payload, write the new seal.  A reader that catches the row mid-write
        sees a seal that does not match its payload and counts the row as
        unwritten -- which is the safe direction -- instead of pairing this
        flip's ``gate_seq`` with the previous store's ``ok`` (:func:`_seal`).
        """
        self._require_flip(f"write_gate_row row={row} wave={wave}")
        off = self._gate_off(row)
        rec = GateRow(row, int(wave) + 1, bool(ok), int(pid or os.getpid()),
                      int(ts_ns or time.time_ns()),
                      GATE_JOINED if ok else GATE_FAILED, self.epoch_hash)
        payload = GATE_PAYLOAD_STRUCT.pack(
            rec.gate_seq, 1 if rec.ok else 0, rec.pid, rec.ts_ns,
            rec.state, rec.epoch_hash, 0,
        )
        struct.pack_into("<Q", self._mm, off + GATE_SEAL_OFF, 0)
        self._mm[off: off + GATE_SEAL_OFF] = payload
        struct.pack_into("<Q", self._mm, off + GATE_SEAL_OFF, _seal(payload))
        return rec

    # ---- matrix rows ----------------------------------------------------

    def _matrix_off(self, row: int) -> int:
        if not 0 <= int(row) < N_RANKS:
            raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
        return MATRIX_OFF + int(row) * MATRIX_ROW_BYTES

    def read_matrix_row(self, row: int) -> MatrixRow:
        off = self._matrix_off(row)
        raw = bytes(self._mm[off: off + MATRIX_ROW_BYTES])
        payload = raw[:MATRIX_SEAL_OFF]
        (seal,) = struct.unpack_from("<Q", raw, MATRIX_SEAL_OFF)
        f = MATRIX_PAYLOAD_STRUCT.unpack(payload)
        return MatrixRow(
            row, f[0:N_RANKS], f[N_RANKS:2 * N_RANKS], f[MX_PLAN_HASH],
            f[MX_EPOCH_HASH], f[MX_PID], bool(f[MX_LOCAL_OK]),
            sealed=(seal == _seal(payload)), oncard_mode=f[MX_ONCARD_MODE],
        )

    def write_matrix_row(self, row: int, send: Sequence[int], recv: Sequence[int],
                         plan_hash: int, *, pid: int = 0,
                         local_ok: bool = True,
                         oncard_mode: int = ONCARD_MODE_UNSTATED) -> MatrixRow:
        self._require_flip(f"write_matrix_row row={row}")
        if len(send) != N_RANKS or len(recv) != N_RANKS:
            raise ValueError(
                f"send/recv must carry one cell per rank ({N_RANKS}), got "
                f"{len(send)}/{len(recv)}"
            )
        off = self._matrix_off(row)
        this_pid = int(pid or os.getpid())
        words = (
            [int(x) for x in send] + [int(x) for x in recv]
            + [int(plan_hash), self.epoch_hash, this_pid, 1 if local_ok else 0,
               int(oncard_mode)]
            + [0] * (MATRIX_PAYLOAD_STRUCT.size // 8 - MX_ONCARD_MODE - 1)
        )
        payload = MATRIX_PAYLOAD_STRUCT.pack(*words)
        struct.pack_into("<Q", self._mm, off + MATRIX_SEAL_OFF, 0)
        self._mm[off: off + MATRIX_SEAL_OFF] = payload
        struct.pack_into("<Q", self._mm, off + MATRIX_SEAL_OFF, _seal(payload))
        return MatrixRow(row, send, recv, plan_hash, self.epoch_hash, this_pid,
                         local_ok, oncard_mode=int(oncard_mode))

    def write_matrix_verdict(self, row: int, local_ok: bool) -> MatrixRow:
        """Re-publish THIS rank's own row with its rank-local Gate-0 verdict.

        Single writer (a rank owns its row), so the read-modify-write is safe;
        the seal is re-made, so a peer never reads the verdict against the
        old payload.
        """
        rec = self.read_matrix_row(row)
        return self.write_matrix_row(row, rec.send, rec.recv, rec.plan_hash,
                                     pid=rec.pid, local_ok=bool(local_ok),
                                     oncard_mode=rec.oncard_mode)


# --------------------------------------------------------------------------
# The acceptance line for the region itself.
# --------------------------------------------------------------------------


def region_line(region: XchgRegion, *, sems: Optional[int] = None) -> str:
    """The grep-able ``WEG2-XCHG-REGION`` line, spec S3's first acceptance.

    ``epoch=`` carries the BOOT NONCE -- the ``<b>`` of the spec's ``<b.f>``.
    **Deviation, stated**: the region is boot-scoped (see the module
    docstring), so it has no flip half to print, and it is created and armed
    before the first flip token exists.  The flip half appears on the
    ``WEG2-XCHG-GATE`` line, which is per flip and prints ``<b.f>`` verbatim.

    ``registered=<n>/6`` is a live read of the six per-rank bytes, never a
    constant: at launch it is ``0/6`` and it reaches ``6/6`` only once every
    rank has called :meth:`XchgRegion.mark_registered` after its
    ``cudaHostRegister`` (S4), which re-emits this line at that moment.  A
    line that printed ``6/6`` unconditionally would be the "unarmed gate
    reading as a passed one" the spec forbids in section 4.2.
    """
    hdr = region.header()
    return (
        f"WEG2-XCHG-REGION epoch={region.boot_nonce} path={region.path} "
        f"slots={N_PAIRS}x{SLOTS_PER_PAIR}x{hdr['slot_bytes'] // MIB}MiB "
        f"bytes={hdr['region_bytes']} registered={region.registered_count()}/{N_RANKS} "
        f"sems={hdr['n_sems'] if sems is None else int(sems)} "
        f"hook_mode={HOOK_MODE_NAMES.get(hdr['hook_mode'], 'unknown')} scope=boot"
    )


# --------------------------------------------------------------------------
# Gate 0 -- the #802 handshake, before a byte moves.
# --------------------------------------------------------------------------


def gate0_publish(region: XchgRegion, row: int, send: Sequence[int],
                  recv: Sequence[int], plan_hash: int,
                  *, local_ok: bool = True,
                  oncard_mode: int = ONCARD_MODE_UNSTATED) -> MatrixRow:
    """Publish this rank's row of the 6x6 byte matrix.  Moves no byte.

    TODO(S6): pass the launcher's ``--weg2-xchg-oncard`` arm as
    ``oncard_mode`` (``ONCARD_MODE_WORD[mode]``).  Until S6 owns that arm every
    rank publishes ``ONCARD_MODE_UNSTATED`` and :func:`gate0_check` says so on
    its own line rather than grading a field nobody fills -- an unarmed gate
    must never read as a passed one.
    """
    return region.write_matrix_row(row, send, recv, plan_hash,
                                   local_ok=local_ok, oncard_mode=oncard_mode)


def _matrix_wait(region: XchgRegion, budget: float, poll_s: float,
                 monotonic: Callable[[], float], proc_root: str,
                 started: float) -> List[MatrixRow]:
    while True:
        rows = [region.read_matrix_row(r) for r in range(N_RANKS)]
        published = [r for r in rows if r.sealed and r.epoch_hash == region.epoch_hash]
        if len(published) == N_RANKS:
            return rows
        if monotonic() - started >= budget:
            missing = [
                f"[group={row_group_rank(r.row)[0]} rank={row_group_rank(r.row)[1]} "
                f"row={r.row} pid={r.pid} sealed={'yes' if r.sealed else 'no'} "
                f"alive_in_proc={'yes' if _pid_alive(r.pid, proc_root) else 'no'}]"
                for r in rows
                if not (r.sealed and r.epoch_hash == region.epoch_hash)
            ]
            raise region._refuse(
                Weg2XchgGateTimeout,
                f"W69 Weg2XchgGateTimeout gate=0 epoch={region.epoch} "
                f"published={len(published)}/{N_RANKS} budget_s={budget} "
                f"waited_s={monotonic() - started:.3f} non-publishers: {' '.join(missing)} "
                f"(denominator: the six SEALED matrix rows carrying epoch_hash="
                f"{region.epoch_hash:#x})",
            )
        time.sleep(poll_s)


def gate0_check(
    region: XchgRegion,
    row: int,
    *,
    front_plan_hash: Optional[int] = None,
    tag_totals: Optional[Mapping[str, int]] = None,
    tms_tag_bytes: Optional[Mapping[str, int]] = None,
    census_unavailable_reason: str = "",
    budget_s: Optional[float] = None,
    proc_root: str = "/proc",
    poll_s: float = GATE_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, object]:
    """Check all 36 cells, every plan hash and every rank's verdict.  W68 on any gap.

    Runs in the RPC preamble, **before any `resume` and before any `pause`**
    (spec section 3.3), so a refusal here costs nothing: neither side has
    mutated a page.

    RANK-LOCAL CHECKS BECOME A GLOBAL REFUSAL, which is the property the first
    cut of this function did not have.  ``front_plan_hash`` and the per-tag
    comparison are arguments THIS rank holds; a rank that raised on them alone
    would leave the other five to proceed and discover it only when the wave-1
    gate expired at the full budget -- after they had moved wave-1 bytes.  So
    the order here is: compute the rank-local verdict, PUBLISH it into this
    rank's own matrix row, wait for all six rows, then refuse if any of the 36
    cells, any plan hash or any of the six verdicts is bad.  Every reader
    reaches the same verdict from the same six rows.

    ``tag_totals`` / ``tms_tag_bytes`` are the **pinned interface to S1/S2**.
    TODO(#1273 S1/S2): S1's plan builder passes its per-tag byte totals as
    ``tag_totals`` and S2 passes the C++ census ``tms_tag_bytes(tag)``
    (``entrypoint.cpp:131``) as ``tms_tag_bytes``; this function owns only the
    comparison and the refusal.

    THE CENSUS IS MANDATORY, NOT OPTIONAL.  The 36-cell check and the plan-hash
    check both compare the ranks against EACH OTHER, so a common-mode
    derivation error -- one shared bug over replicated geometry, which is the
    likely one -- passes them by construction; the spec concedes exactly that
    (*"that proof is a premise, so the handshake is mandatory anyway"*).
    ``tms_tag_bytes`` is the only quantity in Gate 0 that the plan builder did
    not derive, i.e. the only independent oracle.  Omitting it therefore has
    to be a DECLARED omission: pass ``census_unavailable_reason`` and it prints
    as ``tags=skipped(<reason>)``; omit both silently and this refuses.
    """
    if not 0 <= int(row) < N_RANKS:
        raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
    region._require_flip(f"gate0_check row={row}")
    # From here on this view knows which row it owns, so every refusal below
    # can vote ok=False into wave 0's gate row instead of leaving the peers to
    # discover the abort at the fence budget (spec section 3.6).
    region.bind(int(row))
    budget = fence_budget_s() if budget_s is None else float(budget_s)
    started = monotonic()

    have_census = tag_totals is not None and tms_tag_bytes is not None
    if not have_census and not census_unavailable_reason:
        raise region._refuse(
            Weg2XchgPlanDisagree,
            f"W68 Weg2XchgPlanDisagree gate=0 epoch={region.epoch} reader_row={row}: "
            f"tag_totals/tms_tag_bytes were not supplied and no "
            f"census_unavailable_reason was declared.  The 36-cell and plan-hash "
            f"checks compare the ranks against each other, so the C++ census is Gate "
            f"0's only independent oracle -- skipping it by default would arm half a "
            f"gate and print it as a pass",
        )

    # ---- rank-local half, computed BEFORE the row is re-published --------
    local: List[str] = []
    tags_checked = 0
    if have_census:
        for tag in sorted(tag_totals):
            planned = int(tag_totals[tag])
            census = tms_tag_bytes.get(tag)
            tags_checked += 1
            if census is None:
                local.append(f"[tag={tag} planned={planned} tms_tag_bytes=ABSENT]")
            elif int(census) < planned:
                local.append(
                    f"[tag={tag} planned={planned} > tms_tag_bytes={int(census)} "
                    f"delta={planned - int(census)}]"
                )
    own = region.read_matrix_row(row)
    if not (own.sealed and own.epoch_hash == region.epoch_hash):
        raise region._refuse(
            Weg2XchgPlanDisagree,
            f"W68 Weg2XchgPlanDisagree gate=0 epoch={region.epoch} reader_row={row}: "
            f"this rank has not published its own matrix row for this flip "
            f"(sealed={own.sealed} epoch_hash={own.epoch_hash:#x}) -- gate0_publish "
            f"comes first, and a rank that checks a gate it never joined is the "
            f"absent-rank case one step earlier",
        )
    if front_plan_hash is not None and own.plan_hash != int(front_plan_hash):
        local.append(
            f"[plan_hash={own.plan_hash:#x} != front's {int(front_plan_hash):#x}]"
        )
    if local:
        # Tell the other five BEFORE raising: the verdict is in the shared row,
        # so their own gate0_check refuses on the same six rows.
        region.write_matrix_verdict(row, False)

    rows = _matrix_wait(region, budget, poll_s, monotonic, proc_root, started)

    mismatches: List[str] = list(local)
    for a in range(N_RANKS):
        for b in range(N_RANKS):
            sent = rows[a].send[b]
            expected = rows[b].recv[a]
            if sent != expected:
                ga, ra = row_group_rank(a)
                gb, rb = row_group_rank(b)
                mismatches.append(
                    f"[send[{a}][{b}]={sent} != recv[{b}][{a}]={expected} "
                    f"delta={sent - expected} src=group={ga},rank={ra} "
                    f"dst=group={gb},rank={rb}]"
                )
    # THE LOUD HALF OF S4-fix REFUSAL A.  Two co-located ranks that disagree
    # about `ipc` vs `host` fail LATE and confusingly -- a zero IPC handle on
    # one side, a `HostBounce(create=False)` ENOENT on the other, both inside
    # the transport, after the flip has started.  The mode is a per-boot arm,
    # so the disagreement is derivable here, before a byte moves and before any
    # resume, where a refusal costs nothing.  UNSTATED is not a disagreement:
    # until S6 passes the arm down every rank publishes 0, and a gate that
    # refused that would refuse every flip before S6 exists.
    modes = {r.oncard_mode for r in rows}
    stated = modes - {ONCARD_MODE_UNSTATED}
    if len(stated) > 1 or (stated and ONCARD_MODE_UNSTATED in modes):
        mismatches.append(
            "[on-card MODE disagreement: " + " ".join(
                f"row={r.row}:{r.oncard_mode_name}" for r in rows)
            + " -- one arm would export an IPC handle the other never imports "
              "(or open a host bounce file that was never created); it is a "
              "per-boot arm, so this is a launch disagreement, not a race]"
        )
    hashes = {r.plan_hash for r in rows}
    if len(hashes) != 1:
        mismatches.append(
            "[plan_hash differs across ranks: "
            + " ".join(f"row={r.row}:{r.plan_hash:#x}" for r in rows) + "]"
        )
    for r in rows:
        if not r.local_ok:
            group, rank = row_group_rank(r.row)
            mismatches.append(
                f"[group={group} rank={rank} row={r.row} pid={r.pid} published "
                f"local_ok=0: ITS rank-local check (per-tag census or front plan "
                f"hash) failed, and that is this whole gate's refusal, not only "
                f"its own -- its log names the tag]"
            )
    if mismatches:
        raise region._refuse(
            Weg2XchgPlanDisagree,
            f"W68 Weg2XchgPlanDisagree gate=0 epoch={region.epoch} "
            f"reader_row={row} cells=36 verdicts={sum(1 for r in rows if r.local_ok)}/"
            f"{N_RANKS} tags={tags_checked if have_census else 'skipped'} "
            f"mismatches={len(mismatches)}: {' '.join(mismatches)} -- no byte has moved",
        )
    total = sum(sum(r.send) for r in rows)
    return {
        "cells": N_RANKS * N_RANKS,
        "published": len(rows),
        "tags_checked": (
            tags_checked if have_census else f"skipped({census_unavailable_reason})"
        ),
        "verdicts_ok": sum(1 for r in rows if r.local_ok),
        "oncard_mode": ONCARD_MODE_NAMES.get(rows[row].oncard_mode, "unknown"),
        "plan_hash": rows[row].plan_hash,
        "total_bytes": total,
        "waited_s": monotonic() - started,
    }


# --------------------------------------------------------------------------
# The wave gate.
# --------------------------------------------------------------------------


def _describe(rows: Sequence[GateRow], which: Sequence[int], want_seq: int,
              proc_root: str) -> str:
    out = []
    for r in which:
        row = rows[r]
        group, rank = row_group_rank(r)
        out.append(
            f"[group={group} rank={rank} row={r} pid={row.pid} "
            f"wave_seen={row.wave_seen()} wave_wanted={want_seq - 1} ok={row.ok} "
            f"sealed={'yes' if row.sealed else 'no'} "
            f"alive_in_proc={'yes' if _pid_alive(row.pid, proc_root) else 'no'}]"
        )
    return " ".join(out)


def wave_gate(
    region: XchgRegion,
    row: int,
    wave: int,
    ok: bool = True,
    *,
    log: Callable[[str], None],
    budget_s: Optional[float] = None,
    proc_root: str = "/proc",
    poll_s: float = GATE_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, object]:
    """Close wave ``wave`` across all six ranks, or raise W69 naming who did not.

    Spec section 1.3 step 17: each rank writes its own row **after its last
    ``cudaStreamSynchronize``** -- that sync is a syscall, so the payload is
    visibly complete before the row is stored -- and then polls the other five.

    THE DENOMINATOR IS THE ROWS CARRYING THIS FLIP'S EPOCH, and it is printed.
    A row is a join only if its seal validates, its ``epoch_hash`` is this
    flip's, AND its ``gate_seq`` has reached this wave.  Dropping any of the
    three is how a gate passes while a rank is absent: the previous flip left
    all six rows at ``gate_seq=3``, so a count that forgot the epoch would read
    6/6 before anyone had joined anything.

    SIX ROWS ARE NOT SIX RANKS.  On the way out this also proves the six rows
    carry six DISTINCT pids and that every one of them is still in ``/proc``.
    Without that, a rank that wrote its row and then died (OOM, a raise after
    the store) leaves a permanent phantom join: every later gate closes 6/6
    without it, and after gate 1 the source has already unmapped, so the flip
    is committed to the W73 roll-forward by a gate that reported PASS.

    ``log`` is REQUIRED, not defaulted: this line is S3's second acceptance
    criterion, and a default of ``None`` would let a caller silently keep it
    out of the boot log.
    """
    region._require_flip(f"wave_gate row={row} wave={wave}")
    want_seq = int(wave) + 1
    budget = fence_budget_s() if budget_s is None else float(budget_s)
    region.bind(row, wave)
    region.write_gate_row(row, wave, ok)
    started = monotonic()

    def refuse(headline: str, which: Sequence[int], rows: Sequence[GateRow],
               joined: Sequence[int], voted_ok: Sequence[int]) -> Weg2XchgGateTimeout:
        # This rank joined; it is aborting anyway, so it flips its own vote to
        # false before raising. A peer still polling then refuses at once
        # instead of at the fence budget.
        note = region.vote_failure_note()
        return Weg2XchgGateTimeout(
            f"W69 Weg2XchgGateTimeout epoch={region.epoch} wave={wave} "
            f"joined={len(joined)}/{N_RANKS} ok={len(voted_ok)}/{N_RANKS} "
            f"budget_s={budget} waited_s={monotonic() - started:.3f} "
            f"{headline}: {_describe(rows, which, want_seq, proc_root)} "
            f"(denominator: the six rows carrying epoch_hash={region.epoch_hash:#x}) "
            f"{note}"
        )

    while True:
        rows = [region.read_gate_row(r) for r in range(N_RANKS)]
        joined = [r for r in range(N_RANKS)
                  if rows[r].sealed
                  and rows[r].epoch_hash == region.epoch_hash
                  and rows[r].gate_seq >= want_seq]
        voted_ok = [r for r in joined if rows[r].ok]
        failed = [r for r in joined if not rows[r].ok]
        if failed:
            raise refuse(
                "voted_not_ok (at the wave each names in wave_seen; a rank that has "
                "already advanced votes for ITS wave, and a false vote at any wave "
                "aborts the flip)",
                failed, rows, joined, voted_ok,
            )
        dead = [r for r in joined if not _pid_alive(rows[r].pid, proc_root)]
        if dead:
            raise refuse(
                "joined_but_dead (a row whose writer is gone from /proc is not a "
                "join; nobody will produce this rank's wave)",
                dead, rows, joined, voted_ok,
            )
        if len(joined) == N_RANKS:
            break
        if monotonic() - started >= budget:
            absent = [r for r in range(N_RANKS) if r not in joined]
            raise refuse("non_joiners", absent, rows, joined, voted_ok)
        time.sleep(poll_s)

    pids = [rows[r].pid for r in range(N_RANKS)]
    if len(set(pids)) != N_RANKS:
        raise refuse(
            "six_rows_are_not_six_ranks (the rows carry "
            f"{len(set(pids))} distinct pids, not {N_RANKS}: one process wrote "
            "another rank's row, or two rows name the same rank)",
            list(range(N_RANKS)), rows, joined, voted_ok,
        )

    stamps = [rows[r].ts_ns for r in range(N_RANKS)]
    skew_ms = (max(stamps) - min(stamps)) / 1e6
    line = (
        f"WEG2-XCHG-GATE epoch={region.epoch} wave={wave} "
        f"joined={len(joined)}/{N_RANKS} ok={len(voted_ok)}/{N_RANKS} "
        f"skew_ms={skew_ms:.3f} waited_ms={(monotonic() - started) * 1e3:.3f} "
        f"(denominator: the six rows carrying this epoch)"
    )
    log(line)
    return {
        "joined": len(joined),
        "ok": len(voted_ok),
        "skew_ms": skew_ms,
        "waited_s": monotonic() - started,
        "line": line,
    }


# --------------------------------------------------------------------------
# The 24 named POSIX semaphores.
# --------------------------------------------------------------------------

_SEM_FAILED = ctypes.c_void_p(-1).value
_O_CREAT = 0o100  # Linux, all architectures we run on
_O_EXCL = 0o200


def sem_name(boot_nonce: str, pair: int, slot: int, kind: str) -> str:
    """``/weg2-xchg-<boot nonce>-<src>-<dst>-<slot>-{empty,full}``.

    The boot nonce is in the NAME, not only in the region: two boots'
    handshakes then cannot alias, which is the same reason ``credit_epoch``
    composes the boot nonce with the flip index.  Load-bearing and pinned by
    a test -- deleting the nonce from this format string leaves every other
    property of the 24 names (count, uniqueness, creation, unlink) intact,
    so nothing else here would notice.

    The nonce, not the flip epoch: the semaphores belong to the region, which
    is boot-scoped (module docstring), and a per-flip ``sem_open`` x24 on the
    transport's critical path is a cost the flip budget has no term for.
    """
    if kind not in ("empty", "full"):
        raise ValueError(f"kind must be 'empty' or 'full', not {kind!r}")
    src, dst = CROSS_PAIRS[int(pair)]
    return f"/{REGION_PREFIX}{boot_nonce}-{src}-{dst}-{int(slot)}-{kind}"


def all_sem_names(boot_nonce: str) -> List[str]:
    """All 24: six directed pairs x two slots x {empty, full}."""
    return [
        sem_name(boot_nonce, pair, slot, kind)
        for pair in range(N_PAIRS)
        for slot in range(SLOTS_PER_PAIR)
        for kind in ("empty", "full")
    ]


def _libc() -> ctypes.CDLL:
    lib = ctypes.CDLL("libc.so.6", use_errno=True)
    lib.sem_open.restype = ctypes.c_void_p
    lib.sem_unlink.argtypes = [ctypes.c_char_p]
    lib.sem_close.argtypes = [ctypes.c_void_p]
    lib.sem_post.argtypes = [ctypes.c_void_p]
    return lib


def create_semaphores(boot_nonce: str) -> List[str]:
    """Create all 24 with ``empty`` at 1 and ``full`` at 0.

    UNLINK FIRST, THEN ``O_CREAT | O_EXCL``.  POSIX ignores the ``value``
    argument when ``sem_open(O_CREAT)`` finds an existing name -- it returns
    the existing object WITH ITS OLD COUNT.  A surviving ``empty`` at 0 would
    then make the first producer of the boot block until the 120 s fence
    budget instead of refusing at launch, and no line anywhere would say the
    initial values were not applied.  ``O_EXCL`` makes "this is a fresh
    handshake" checkable rather than assumed, and the ``sem_unlink`` before it
    is what makes the fresh creation possible after a crash (it detaches the
    name; any process still holding the old object keeps it and dies with it).

    Called only by the launcher, before either group starts, and only after
    the residue sweep -- which refuses the boot outright while another
    ``launch_server`` is alive, so the unlink here can never detach a live
    boot's handshake.  S4 owns ``sem_wait``/``sem_post``/``sem_timedwait``;
    this slice creates and destroys the names and nothing else.
    """
    lib = _libc()
    names = all_sem_names(boot_nonce)
    for name in names:
        lib.sem_unlink(name.encode("ascii"))
    made: List[str] = []
    for name in names:
        initial = 1 if name.endswith("-empty") else 0
        handle = lib.sem_open(name.encode("ascii"), _O_CREAT | _O_EXCL, 0o600, initial)
        if handle in (None, 0, _SEM_FAILED):
            err = ctypes.get_errno()
            unlink_semaphores(boot_nonce)
            raise OSError(err, f"sem_open({name}) failed: {os.strerror(err)}")
        lib.sem_close(ctypes.c_void_p(handle))
        made.append(name)
    return made


def unlink_semaphores(boot_nonce: str) -> int:
    """``sem_unlink`` all 24.  Idempotent: an absent name is not an error.

    Called at launch AND at teardown (spec section 3.8): teardown alone leaves
    a crashed boot's names behind, which is the case the boot nonce in the
    name closes.

    ``ENOENT`` is the only tolerated failure.  A ``-1`` return with ``errno``
    unset is an anomaly, not a success: reading it as one would report
    "24 unlinked" for names that are still there.
    """
    lib = _libc()
    gone = 0
    for name in all_sem_names(boot_nonce):
        ctypes.set_errno(0)
        if lib.sem_unlink(name.encode("ascii")) == 0:
            gone += 1
            continue
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(
                err,
                f"sem_unlink({name}) returned -1 with errno={err} "
                f"({os.strerror(err) if err else 'errno unset'}) -- the name may "
                f"still be there and this count would have lied about it",
            )
    return gone


# --------------------------------------------------------------------------
# Launch-side entry points (the launcher calls these; see launcher.py).
# --------------------------------------------------------------------------


def prepare_region(
    boot_nonce: str,
    *,
    shm_root: str = SHM_ROOT,
    hook_mode: int = HOOK_MODE_UNKNOWN,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Create the region and its semaphores; return the env both groups need.

    Returns ``{"path", "boot", "env", "sems", "line"}``.  The two env vars are
    published to BOTH ``launch_server`` process trees -- one region, six
    ranks, and every rank finds it by the same two names.  ONCE PER BOOT: the
    flip token is not known here and does not belong here (module docstring).

    If the semaphores cannot be created, the region file is removed again.
    Leaving 385 MiB of region behind with no handshake would mean the next
    launch meets a region ``create``'s ``O_EXCL`` refuses, for a boot that
    never started.
    """
    region = XchgRegion.create(boot_nonce, shm_root=shm_root, hook_mode=hook_mode)
    try:
        sems = create_semaphores(boot_nonce)
        region.set_sem_count(len(sems))
        line = region_line(region)
        if log is not None:
            log(line)
        return {
            "path": region.path,
            "boot": str(boot_nonce),
            "env": {ENV_REGION_PATH: region.path, ENV_REGION_BOOT: str(boot_nonce)},
            "sems": len(sems),
            "line": line,
        }
    except BaseException:
        region.close()
        teardown_region(boot_nonce, shm_root=shm_root)
        raise
    finally:
        if region._fd >= 0:
            region.close()


def teardown_region(boot_nonce: str, *, shm_root: str = SHM_ROOT,
                    log: Optional[Callable[[str], None]] = None) -> Dict[str, int]:
    """``sem_unlink`` the 24 names and remove the region directory."""
    sems = unlink_semaphores(boot_nonce)
    path = region_path(boot_nonce, shm_root)
    removed = 0
    try:
        os.unlink(path)
        removed = 1
    except FileNotFoundError:
        pass
    try:
        os.rmdir(region_dir(boot_nonce, shm_root))
    except OSError:
        pass
    if log is not None:
        log(f"WEG2-XCHG-TEARDOWN boot={boot_nonce} sems_unlinked={sems}/24 "
            f"region_removed={removed}")
    return {"sems": sems, "region": removed}
