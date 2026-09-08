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

THREE PROPERTIES THIS FILE EXISTS TO HOLD, each of which is a measured
failure of a predecessor mechanism rather than a precaution:

* **A gate that can pass while a rank is absent is not a gate** (S3's
  stop-loss).  Every joined-row count in this file filters on *this epoch's*
  hash, carried in the row itself, and every emitted count prints its
  denominator.  A row left behind by the previous flip has the previous
  epoch's hash and can never be counted as a join -- which is the same rule
  ``credit_epoch`` (:mod:`sglang.srt.managers.weg2_memory_saver`) was given
  after boot weg2rg2 left three terminal credit counters on ``/dev/shm``,
  one carrying ``{"epoch": 12, ... "leg_complete": true}``, that a later
  boot's flip 12 would have read as its own funding.
* **Byte counts are agreed before a byte moves** (#802).  ``c26d28172106``
  killed an instance for a corruption that had not happened, because
  ``_dist_exchange`` derived send/recv byte counts independently on each rank
  and never handshook them.  Gate 0 is that handshake: every rank publishes
  BOTH what it will send and what it expects to receive, and every rank
  checks all 36 cells.
* **A buffer's size never stands for its filled bytes** (#802 rule 2).
  ``bytes_filled`` is written by the producer *after* its own
  ``cudaStreamSynchronize`` and is the only quantity a consumer may copy.
  This module owns the field and its state machine; S4 owns the copy that
  reads it (W54).

THE TWO W-CODES ALLOCATED HERE are W52 and W53 (spec section 7).  Both were
free at the tree of record: the census ``test_weg2_wcode_uniqueness_1263.py``
performs over its own ROOTS returns W50 as the highest assigned code.

LAYOUT (spec S3, with the two deviations stated where they occur)::

    /dev/shm/weg2-xchg-<epoch>/xchg.bin
      [0        ) header : magic, version, epoch, n_ranks, slot_bytes,
                           hook_mode, registered bitmap  + the 12 slots
      [4 KiB    ) gate   : 6 rows x 64 B
      [8 KiB    ) matrix : 6 rows x 128 B  (send row AND recv row)
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
HEADER_FIELDS_BYTES = 128
EPOCH_STR_OFF = HEADER_OFF + HEADER_FIELDS_BYTES
EPOCH_STR_BYTES = 128
SLOTS_OFF = 256
SLOT_STRUCT = struct.Struct("<IIQQQQii")  # 48 B, the spec's XchgSlot, in order
SLOT_RECORD_BYTES = SLOT_STRUCT.size
N_SLOTS = N_PAIRS * SLOTS_PER_PAIR

GATE_OFF = 4 * KIB
#: 64 B per row, because spec section 3.2 says "six 64-byte-aligned shm rows"
#: -- one cache line per rank, so one rank's store never invalidates another's.
GATE_ROW_BYTES = 64
GATE_ROW_STRUCT = struct.Struct("<8Q")

MATRIX_OFF = 8 * KIB
#: **DEVIATION from the spec's "6 rows x 6 u64", with its reason.** The check
#: the spec mandates is ``send[a][b] == recv[b][a]`` over all 36 cells -- and
#: that is not computable from a send-only matrix, because nothing would carry
#: ``recv``.  Each rank therefore publishes BOTH of its own vectors: what it
#: will send to each peer, and what it expects to receive from each peer.  The
#: row is 128 B so both vectors plus the plan hash, the epoch hash and the pid
#: fit in two cache lines.
MATRIX_ROW_BYTES = 128
MATRIX_ROW_STRUCT = struct.Struct("<16Q")

DIR_OFF = 64 * KIB
DATA_OFF = 1 * MIB
DATA_BYTES = N_PAIRS * SLOTS_PER_PAIR * SLOT_BYTES
REGION_BYTES = DATA_OFF + DATA_BYTES

XCHG_MAGIC = int.from_bytes(b"WEG2XCHG", "little")
XCHG_VERSION = 1

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
ENV_REGION_EPOCH = "SGLANG_WEG2_XCHG_EPOCH"

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
    """W52: Gate 0 found the two sides of a byte count disagreeing.

    Raised on EVERY reader, never only on the rank that happens to notice --
    #802's rule is that the disagreement is a property of the pair, so both
    halves must stop.  The message names the cell, both values and their
    difference, because "plan mismatch" without the cell is a postmortem
    nobody can start from.
    """


class Weg2XchgGateTimeout(RuntimeError):
    """W53: a rank did not join a gate, or joined it with ``ok=False``.

    Names every non-joiner by group, rank, row, pid, the wave it was last
    seen at, the wave it was needed at, and whether its pid is still in
    ``/proc`` -- the poll knows all of that, which is precisely why spec
    section 3.2 refuses a gloo ``monitored_barrier`` for this gate.
    """


# --------------------------------------------------------------------------
# Identity helpers.
# --------------------------------------------------------------------------


def epoch_hash(epoch: str) -> int:
    """A stable 64-bit hash of the flip epoch token ``"<boot>.<flip>"``.

    ``hash()`` is NOT usable: PYTHONHASHSEED randomises it per process, and
    the six ranks are six processes -- they would each stamp a different
    number into shared memory and every epoch filter in this file would
    reject every peer.  blake2b is stable across processes and boots.

    Never zero: zero is the zeroed region, i.e. "nobody wrote here", and an
    epoch that collided with it would make an untouched row indistinguishable
    from a joined one.
    """
    digest = hashlib.blake2b(str(epoch).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value or 1


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


def region_dir(epoch: str, shm_root: str = SHM_ROOT) -> str:
    return os.path.join(shm_root, f"{REGION_PREFIX}{epoch}")


def region_path(epoch: str, shm_root: str = SHM_ROOT) -> str:
    return os.path.join(region_dir(epoch, shm_root), REGION_FILE)


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

    __slots__ = ("row", "gate_seq", "ok", "pid", "ts_ns", "state", "epoch_hash")

    def __init__(self, row, gate_seq, ok, pid, ts_ns, state, eh):
        self.row = int(row)
        self.gate_seq = int(gate_seq)
        self.ok = bool(ok)
        self.pid = int(pid)
        self.ts_ns = int(ts_ns)
        self.state = int(state)
        self.epoch_hash = int(eh)

    @property
    def group_rank(self) -> Tuple[str, int]:
        return row_group_rank(self.row)

    def wave_seen(self) -> int:
        """The wave this row last joined, or -1 for a row nobody ever wrote."""
        return self.gate_seq - 1


class MatrixRow:
    """One rank's published send and recv vectors, plus its plan hash."""

    __slots__ = ("row", "send", "recv", "plan_hash", "epoch_hash", "pid")

    def __init__(self, row, send, recv, plan_hash, eh, pid):
        self.row = int(row)
        self.send = tuple(int(x) for x in send)
        self.recv = tuple(int(x) for x in recv)
        self.plan_hash = int(plan_hash)
        self.epoch_hash = int(eh)
        self.pid = int(pid)


# --------------------------------------------------------------------------
# The region.
# --------------------------------------------------------------------------


class XchgRegion:
    """The mapped exchange region.  Every rank of both groups maps this file.

    The object is a VIEW, never a cache: every read goes to the mapping, so a
    peer's store is visible on the next read.  ``sem_post``/``sem_wait`` (S4)
    and the syscall each rank issues before writing its gate row are the
    synchronisation; nothing here needs a fence of its own.
    """

    def __init__(self, path: str, mm: mmap.mmap, fd: int, epoch: str):
        self.path = path
        self._mm = mm
        self._fd = fd
        self.epoch = epoch
        self.epoch_hash = epoch_hash(epoch)

    # ---- lifecycle ------------------------------------------------------

    @classmethod
    def create(
        cls,
        epoch: str,
        *,
        path: str = "",
        shm_root: str = SHM_ROOT,
        hook_mode: int = HOOK_MODE_UNKNOWN,
        slot_bytes: int = SLOT_BYTES,
    ) -> "XchgRegion":
        """Create (or re-create) the region for ``epoch`` and zero its gates.

        ``ftruncate`` to the full :data:`REGION_BYTES`: on tmpfs the file is
        sparse until written, so the 385 MiB is charged page by page as the
        transport touches it, and the gate/matrix/header cost ~72 KiB.
        """
        target = path or region_path(epoch, shm_root)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd = os.open(target, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, REGION_BYTES)
            mm = mmap.mmap(fd, REGION_BYTES, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(fd)
            raise
        region = cls(target, mm, fd, str(epoch))
        # Zero everything a stale re-create could leave behind. The DATA area
        # is deliberately NOT zeroed (385 MiB of pointless page faults); a
        # slot's content is only ever read through its own header record,
        # whose epoch hash is checked (see `claim_produced`).
        mm[0:DATA_OFF] = b"\x00" * DATA_OFF
        header = [
            XCHG_MAGIC, XCHG_VERSION, region.epoch_hash, N_RANKS,
            int(slot_bytes), int(hook_mode), 0, REGION_BYTES,
            N_PAIRS, SLOTS_PER_PAIR, time.time_ns(), os.getpid(),
            0, 0, 0, 0,
        ]
        struct.pack_into("<16Q", mm, HEADER_OFF, *header)
        raw = str(epoch).encode("utf-8")[: EPOCH_STR_BYTES - 1]
        mm[EPOCH_STR_OFF: EPOCH_STR_OFF + EPOCH_STR_BYTES] = raw.ljust(EPOCH_STR_BYTES, b"\x00")
        for index in range(N_SLOTS):
            region._write_slot_record(index, XchgSlot(
                SLOT_FREE, index // SLOTS_PER_PAIR, 0, 0, 0, 0, 0, 0, index))
        return region

    @classmethod
    def open(cls, path: str, *, expect_epoch: str = "") -> "XchgRegion":
        """Map an existing region.  Refuses a foreign or stale file by name."""
        fd = os.open(path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, REGION_BYTES, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(fd)
            raise
        magic, version = struct.unpack_from("<2Q", mm, HEADER_OFF)
        if magic != XCHG_MAGIC:
            mm.close()
            os.close(fd)
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree path={path}: magic {magic:#x} is not "
                f"{XCHG_MAGIC:#x} -- this is not an exchange region"
            )
        if version != XCHG_VERSION:
            mm.close()
            os.close(fd)
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree path={path}: region version {version} != "
                f"{XCHG_VERSION} -- two builds of the exchange in one boot"
            )
        raw = bytes(mm[EPOCH_STR_OFF: EPOCH_STR_OFF + EPOCH_STR_BYTES])
        epoch = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
        if expect_epoch and epoch != str(expect_epoch):
            mm.close()
            os.close(fd)
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree path={path}: region carries epoch "
                f"{epoch!r}, this rank was launched for {expect_epoch!r} -- a "
                f"previous boot's region, never adopted as this one's"
            )
        return cls(path, mm, fd, epoch)

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

    # ---- header ---------------------------------------------------------

    def header(self) -> Dict[str, int]:
        f = struct.unpack_from("<16Q", self._mm, HEADER_OFF)
        return {
            "magic": f[0], "version": f[1], "epoch_hash": f[2], "n_ranks": f[3],
            "slot_bytes": f[4], "hook_mode": f[5], "registered_bits": f[6],
            "region_bytes": f[7], "n_pairs": f[8], "slots_per_pair": f[9],
            "created_ns": f[10], "creator_pid": f[11],
        }

    def mark_registered(self, row: int) -> int:
        """Record that one rank has ``cudaHostRegister``ed this region (S4).

        A bitmap rather than a counter: a rank that registers twice must not
        make the denominator read 7/6, and a rank that never registers must be
        namable, not merely missing from a total.
        """
        bits = self.header()["registered_bits"] | (1 << int(row))
        struct.pack_into("<Q", self._mm, HEADER_OFF + 6 * 8, bits)
        return bits

    def registered_count(self) -> int:
        return bin(self.header()["registered_bits"]).count("1")

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
        SLOT_STRUCT.pack_into(
            self._mm, self._slot_off(index),
            rec.state, rec.pair_id, rec.epoch_hash, rec.seq,
            rec.bytes_filled, rec.checksum, rec.producer_pid, rec.consumer_pid,
        )

    def begin_fill(self, pair: int, slot: int, seq: int, *, producer_pid: int = 0) -> XchgSlot:
        """Claim a slot for this flip and mark it FILLING.  No bytes yet."""
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
        """
        index = self.slot_index(pair, slot)
        rec = self.read_slot(pair, slot)
        if rec.state != SLOT_FILLING or rec.epoch_hash != self.epoch_hash:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree publish pair={pair} slot={slot}: the slot is "
                f"{rec.state_name} at epoch_hash={rec.epoch_hash:#x}, expected FILLING at "
                f"{self.epoch_hash:#x} -- a publish onto a slot this flip never claimed"
            )
        if int(bytes_filled) > SLOT_BYTES:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree publish pair={pair} slot={slot}: "
                f"bytes_filled={bytes_filled} exceeds slot_bytes={SLOT_BYTES}"
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

        * **PRODUCED at this epoch** -> returned, state DRAINING, and it does
          NOT matter whether the producer is still alive.  A producer that
          published and then died has already delivered; killing the consumer
          over it would destroy bytes that are sitting in shared memory
          (judges' graft 5).
        * **any state at a FOREIGN epoch** -> the record is zeroed and ``None``
          is returned.  This is ``host_ring.cpp:285-291``'s rule ported to
          slots: a previous boot's or a previous flip's leftovers are never
          read as this flip's funding.
        * **FILLING at this epoch whose producer is gone from /proc** -> W53.
          Nobody will ever post this slot, and a bounded wait that ends in a
          timeout would name the wait rather than the death.
        """
        rec = self.read_slot(pair, slot)
        if rec.epoch_hash != self.epoch_hash:
            if rec.state != SLOT_FREE or rec.bytes_filled:
                self._write_slot_record(rec.index, XchgSlot(
                    SLOT_FREE, pair, 0, 0, 0, 0, 0, 0, rec.index))
            return None
        if rec.state == SLOT_FILLING:
            alive = os.path.isdir(os.path.join(proc_root, str(rec.producer_pid)))
            if not alive:
                raise Weg2XchgGateTimeout(
                    f"W53 Weg2XchgGateTimeout slot pair={pair} slot={slot} state=FILLING "
                    f"epoch={self.epoch} producer_pid={rec.producer_pid} "
                    f"alive_in_proc=no seq={rec.seq} bytes_filled={rec.bytes_filled} -- the "
                    f"producer died mid-fill; these bytes will never be posted"
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
        index = self.slot_index(pair, slot)
        self._write_slot_record(index, XchgSlot(
            SLOT_FREE, pair, self.epoch_hash, 0, 0, 0, 0, 0, index))

    # ---- gate rows ------------------------------------------------------

    def _gate_off(self, row: int) -> int:
        if not 0 <= int(row) < N_RANKS:
            raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
        return GATE_OFF + int(row) * GATE_ROW_BYTES

    def read_gate_row(self, row: int) -> GateRow:
        f = GATE_ROW_STRUCT.unpack_from(self._mm, self._gate_off(row))
        return GateRow(row, f[0], bool(f[1]), f[2], f[3], f[4], f[5])

    def write_gate_row(self, row: int, wave: int, ok: bool, *, pid: int = 0,
                       ts_ns: int = 0) -> GateRow:
        """Publish this rank's vote for ``wave``.

        The row carries THIS flip's epoch hash, which is what makes a row left
        behind by the previous flip uncountable: it has the previous epoch's
        hash, and every joined-count in this module filters on the hash.
        """
        rec = GateRow(row, int(wave) + 1, bool(ok), int(pid or os.getpid()),
                      int(ts_ns or time.time_ns()),
                      GATE_JOINED if ok else GATE_FAILED, self.epoch_hash)
        GATE_ROW_STRUCT.pack_into(
            self._mm, self._gate_off(row),
            rec.gate_seq, 1 if rec.ok else 0, rec.pid, rec.ts_ns,
            rec.state, rec.epoch_hash, 0, 0,
        )
        return rec

    # ---- matrix rows ----------------------------------------------------

    def _matrix_off(self, row: int) -> int:
        if not 0 <= int(row) < N_RANKS:
            raise ValueError(f"row must be 0..{N_RANKS - 1}, not {row!r}")
        return MATRIX_OFF + int(row) * MATRIX_ROW_BYTES

    def read_matrix_row(self, row: int) -> MatrixRow:
        f = MATRIX_ROW_STRUCT.unpack_from(self._mm, self._matrix_off(row))
        return MatrixRow(row, f[0:6], f[6:12], f[12], f[13], f[14])

    def write_matrix_row(self, row: int, send: Sequence[int], recv: Sequence[int],
                         plan_hash: int, *, pid: int = 0) -> MatrixRow:
        if len(send) != N_RANKS or len(recv) != N_RANKS:
            raise ValueError(
                f"send/recv must carry one cell per rank ({N_RANKS}), got "
                f"{len(send)}/{len(recv)}"
            )
        values = (list(int(x) for x in send) + list(int(x) for x in recv)
                  + [int(plan_hash), self.epoch_hash, int(pid or os.getpid()), 0])
        MATRIX_ROW_STRUCT.pack_into(self._mm, self._matrix_off(row), *values)
        return MatrixRow(row, send, recv, plan_hash, self.epoch_hash, pid or os.getpid())


# --------------------------------------------------------------------------
# The acceptance line for the region itself.
# --------------------------------------------------------------------------


def region_line(region: XchgRegion, *, sems: int = 0) -> str:
    """The grep-able ``WEG2-XCHG-REGION`` line, spec S3's first acceptance.

    ``registered=<n>/6`` is a live read of the header bitmap, never a constant:
    at launch it is ``0/6`` and it reaches ``6/6`` only once every rank has
    called :meth:`XchgRegion.mark_registered` after its
    ``cudaHostRegister`` (S4).  A line that printed ``6/6`` unconditionally
    would be the "unarmed gate reading as a passed one" the spec forbids in
    section 4.2.
    """
    hdr = region.header()
    return (
        f"WEG2-XCHG-REGION epoch={region.epoch} path={region.path} "
        f"slots={N_PAIRS}x{SLOTS_PER_PAIR}x{hdr['slot_bytes'] // MIB}MiB "
        f"bytes={hdr['region_bytes']} registered={region.registered_count()}/{N_RANKS} "
        f"sems={sems} hook_mode={HOOK_MODE_NAMES.get(hdr['hook_mode'], 'unknown')}"
    )


# --------------------------------------------------------------------------
# Gate 0 -- the #802 handshake, before a byte moves.
# --------------------------------------------------------------------------


def gate0_publish(region: XchgRegion, row: int, send: Sequence[int],
                  recv: Sequence[int], plan_hash: int) -> MatrixRow:
    """Publish this rank's row of the 6x6 byte matrix.  Moves no byte."""
    return region.write_matrix_row(row, send, recv, plan_hash)


def gate0_check(
    region: XchgRegion,
    row: int,
    *,
    front_plan_hash: Optional[int] = None,
    tag_totals: Optional[Mapping[str, int]] = None,
    tms_tag_bytes: Optional[Mapping[str, int]] = None,
    budget_s: Optional[float] = None,
    proc_root: str = "/proc",
    poll_s: float = GATE_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, object]:
    """Check all 36 cells and every rank's plan hash.  Raise W52 on any gap.

    Runs in the RPC preamble, **before any `resume` and before any `pause`**
    (spec section 3.3), so a refusal here costs nothing: neither side has
    mutated a page.

    ``tag_totals`` / ``tms_tag_bytes`` are the **pinned interface to S1/S2**.
    TODO(#1273 S1/S2): S1's plan builder passes its per-tag byte totals as
    ``tag_totals`` and S2 passes the C++ census ``tms_tag_bytes(tag)``
    (``entrypoint.cpp:131``) as ``tms_tag_bytes``; this function owns only the
    comparison and the refusal.  Both default to ``None`` -- absent, which is
    printed as ``tags=skipped``, never silently as agreement.
    """
    budget = fence_budget_s() if budget_s is None else float(budget_s)
    started = monotonic()
    while True:
        rows = [region.read_matrix_row(r) for r in range(N_RANKS)]
        published = [r for r in rows if r.epoch_hash == region.epoch_hash]
        if len(published) == N_RANKS:
            break
        if monotonic() - started >= budget:
            missing = [
                f"[group={row_group_rank(r.row)[0]} rank={row_group_rank(r.row)[1]} "
                f"row={r.row} pid={r.pid} alive_in_proc="
                f"{'yes' if r.pid and os.path.isdir(os.path.join(proc_root, str(r.pid))) else 'no'}]"
                for r in rows if r.epoch_hash != region.epoch_hash
            ]
            raise Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout gate=0 epoch={region.epoch} "
                f"published={len(published)}/{N_RANKS} budget_s={budget} "
                f"waited_s={monotonic() - started:.3f} non-publishers: {' '.join(missing)} "
                f"(denominator: the six matrix rows carrying epoch_hash="
                f"{region.epoch_hash:#x})"
            )
        time.sleep(poll_s)

    mismatches: List[str] = []
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
    hashes = {r.plan_hash for r in rows}
    if len(hashes) != 1:
        mismatches.append(
            "[plan_hash differs across ranks: "
            + " ".join(f"row={r.row}:{r.plan_hash:#x}" for r in rows) + "]"
        )
    elif front_plan_hash is not None and rows[row].plan_hash != int(front_plan_hash):
        mismatches.append(
            f"[plan_hash={rows[row].plan_hash:#x} != front's {int(front_plan_hash):#x}]"
        )

    tags_checked = 0
    if tag_totals is not None and tms_tag_bytes is not None:
        for tag in sorted(tag_totals):
            planned = int(tag_totals[tag])
            census = tms_tag_bytes.get(tag)
            tags_checked += 1
            if census is None:
                mismatches.append(f"[tag={tag} planned={planned} tms_tag_bytes=ABSENT]")
            elif int(census) < planned:
                mismatches.append(
                    f"[tag={tag} planned={planned} > tms_tag_bytes={int(census)} "
                    f"delta={planned - int(census)}]"
                )
    if mismatches:
        raise Weg2XchgPlanDisagree(
            f"W52 Weg2XchgPlanDisagree gate=0 epoch={region.epoch} "
            f"reader_row={row} cells=36 tags={tags_checked if tags_checked else 'skipped'} "
            f"mismatches={len(mismatches)}: {' '.join(mismatches)} -- no byte has moved"
        )
    total = sum(sum(r.send) for r in rows)
    return {
        "cells": N_RANKS * N_RANKS,
        "published": len(rows),
        "tags_checked": tags_checked if (tag_totals is not None and tms_tag_bytes is not None) else "skipped",
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
        alive = "yes" if row.pid and os.path.isdir(os.path.join(proc_root, str(row.pid))) else "no"
        out.append(
            f"[group={group} rank={rank} row={r} pid={row.pid} "
            f"wave_seen={row.wave_seen()} wave_wanted={want_seq - 1} ok={row.ok} "
            f"alive_in_proc={alive}]"
        )
    return " ".join(out)


def wave_gate(
    region: XchgRegion,
    row: int,
    wave: int,
    ok: bool = True,
    *,
    budget_s: Optional[float] = None,
    proc_root: str = "/proc",
    poll_s: float = GATE_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Close wave ``wave`` across all six ranks, or raise W53 naming who did not.

    Spec section 1.3 step 17: each rank writes its own row **after its last
    ``cudaStreamSynchronize``** -- that sync is a syscall, so the payload is
    visibly complete before the row is stored -- and then polls the other five.

    THE DENOMINATOR IS THE ROWS CARRYING THIS EPOCH, and it is printed.  A row
    is a join only if its ``epoch_hash`` is this flip's AND its ``gate_seq``
    has reached this wave.  Dropping either half is how a gate passes while a
    rank is absent: the previous flip left all six rows at ``gate_seq=3``, so
    an unfiltered count would read 6/6 before anyone had joined anything.
    """
    want_seq = int(wave) + 1
    budget = fence_budget_s() if budget_s is None else float(budget_s)
    region.write_gate_row(row, wave, ok)
    started = monotonic()
    while True:
        rows = [region.read_gate_row(r) for r in range(N_RANKS)]
        joined = [r for r in range(N_RANKS)
                  if rows[r].epoch_hash == region.epoch_hash
                  and rows[r].gate_seq >= want_seq]
        voted_ok = [r for r in joined if rows[r].ok]
        failed = [r for r in joined if not rows[r].ok]
        if failed:
            raise Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout epoch={region.epoch} wave={wave} "
                f"joined={len(joined)}/{N_RANKS} ok={len(voted_ok)}/{N_RANKS} "
                f"budget_s={budget} waited_s={monotonic() - started:.3f} "
                f"voted_not_ok: {_describe(rows, failed, want_seq, proc_root)} "
                f"(denominator: the six rows carrying epoch_hash={region.epoch_hash:#x})"
            )
        if len(joined) == N_RANKS:
            break
        if monotonic() - started >= budget:
            absent = [r for r in range(N_RANKS) if r not in joined]
            raise Weg2XchgGateTimeout(
                f"W53 Weg2XchgGateTimeout epoch={region.epoch} wave={wave} "
                f"joined={len(joined)}/{N_RANKS} ok={len(voted_ok)}/{N_RANKS} "
                f"budget_s={budget} waited_s={monotonic() - started:.3f} "
                f"non_joiners: {_describe(rows, absent, want_seq, proc_root)} "
                f"(denominator: the six rows carrying epoch_hash={region.epoch_hash:#x})"
            )
        time.sleep(poll_s)

    stamps = [rows[r].ts_ns for r in range(N_RANKS)]
    skew_ms = (max(stamps) - min(stamps)) / 1e6
    line = (
        f"WEG2-XCHG-GATE epoch={region.epoch} wave={wave} "
        f"joined={len(joined)}/{N_RANKS} ok={len(voted_ok)}/{N_RANKS} "
        f"skew_ms={skew_ms:.3f} waited_ms={(monotonic() - started) * 1e3:.3f} "
        f"(denominator: the six rows carrying this epoch)"
    )
    if log is not None:
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


def sem_name(epoch: str, pair: int, slot: int, kind: str) -> str:
    """``/weg2-xchg-<epoch>-<src>-<dst>-<slot>-{empty,full}``.

    The epoch is in the NAME, not only in the region: two boots' semaphores
    then cannot alias, which is the same reason ``credit_epoch`` composes the
    boot nonce with the flip index.
    """
    if kind not in ("empty", "full"):
        raise ValueError(f"kind must be 'empty' or 'full', not {kind!r}")
    src, dst = CROSS_PAIRS[int(pair)]
    return f"/{REGION_PREFIX}{epoch}-{src}-{dst}-{int(slot)}-{kind}"


def all_sem_names(epoch: str) -> List[str]:
    """All 24: six directed pairs x two slots x {empty, full}."""
    return [
        sem_name(epoch, pair, slot, kind)
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


def create_semaphores(epoch: str) -> List[str]:
    """Create all 24 with ``O_CREAT``, ``empty`` at 1 and ``full`` at 0.

    ``sem_open(O_CREAT)`` is atomic, so there is no init race and no
    ``pthread_mutex`` in shared memory to make robust (spec S3).  Any rank may
    call this; the launcher calls it first so a rank never has to.
    """
    lib = _libc()
    made: List[str] = []
    for name in all_sem_names(epoch):
        initial = 1 if name.endswith("-empty") else 0
        handle = lib.sem_open(name.encode("ascii"), _O_CREAT, 0o600, initial)
        if handle in (None, 0, _SEM_FAILED):
            err = ctypes.get_errno()
            unlink_semaphores(epoch)
            raise OSError(err, f"sem_open({name}) failed: {os.strerror(err)}")
        lib.sem_close(ctypes.c_void_p(handle))
        made.append(name)
    return made


def unlink_semaphores(epoch: str) -> int:
    """``sem_unlink`` all 24.  Idempotent: an absent name is not an error.

    Called at launch AND at teardown (spec section 3.8): teardown alone leaves
    a crashed boot's names behind, which is the case the epoch in the name
    closes.
    """
    lib = _libc()
    gone = 0
    for name in all_sem_names(epoch):
        if lib.sem_unlink(name.encode("ascii")) == 0:
            gone += 1
        else:
            err = ctypes.get_errno()
            if err not in (errno.ENOENT, 0):
                raise OSError(err, f"sem_unlink({name}) failed: {os.strerror(err)}")
    return gone


# --------------------------------------------------------------------------
# Launch-side entry points (the launcher calls these; see launcher.py).
# --------------------------------------------------------------------------


def prepare_region(
    epoch: str,
    *,
    shm_root: str = SHM_ROOT,
    hook_mode: int = HOOK_MODE_UNKNOWN,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Create the region and its semaphores; return the env both groups need.

    Returns ``{"path", "epoch", "env", "sems", "line"}``.  The two env vars are
    published to BOTH ``launch_server`` process trees -- one region, six
    ranks, and every rank finds it by the same two names.
    """
    region = XchgRegion.create(epoch, shm_root=shm_root, hook_mode=hook_mode)
    try:
        sems = create_semaphores(epoch)
        line = region_line(region, sems=len(sems))
        if log is not None:
            log(line)
        return {
            "path": region.path,
            "epoch": str(epoch),
            "env": {ENV_REGION_PATH: region.path, ENV_REGION_EPOCH: str(epoch)},
            "sems": len(sems),
            "line": line,
        }
    finally:
        region.close()


def teardown_region(epoch: str, *, shm_root: str = SHM_ROOT,
                    log: Optional[Callable[[str], None]] = None) -> Dict[str, int]:
    """``sem_unlink`` the 24 names and remove the region directory."""
    sems = unlink_semaphores(epoch)
    path = region_path(epoch, shm_root)
    removed = 0
    try:
        os.unlink(path)
        removed = 1
    except FileNotFoundError:
        pass
    try:
        os.rmdir(region_dir(epoch, shm_root))
    except OSError:
        pass
    if log is not None:
        log(f"WEG2-XCHG-TEARDOWN epoch={epoch} sems_unlinked={sems}/24 region_removed={removed}")
    return {"sems": sems, "region": removed}
