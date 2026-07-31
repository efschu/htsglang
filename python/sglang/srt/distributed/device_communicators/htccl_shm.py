# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the shvllm fork (branch feature/htccl),
# vllm/distributed/device_communicators/htccl_shm.py
"""HTCCL shared-memory transport.

Replicates the data movement of NCCL's no-P2P SHM path with no vendor
library involved: every rank owns a slot in one POSIX shared-memory
segment, registers the mapping as pinned with its own runtime
(``cudaHostRegister`` / ``hipHostRegister``), DMA-copies its tensor
into the slot, and reads the peers' slots straight back over PCIe. The
reduction happens on the GPU in the tensor's own precision.

Synchronization is a lock-free sequence-counter spin (same idea as
``shm_broadcast``): rank r publishes "my slot holds data for op #seq"
by writing seq to its counter; peers spin until every counter reaches
seq. Two counter phases per op (publish, consumed) prevent a fast rank
from overwriting a slot the slow rank is still reading.

Single node only — exactly the scope of a mixed NVIDIA+AMD box.
"""

import ctypes
import ctypes.util
import logging
from multiprocessing import shared_memory

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators import htccl_liveness

logger = logging.getLogger(__name__)


_ONCE_SEEN: set = set()


def _info_once(msg: str, *args) -> None:
    """sglang's logger has no `info_once`; keep the vLLM call sites intact."""
    if msg in _ONCE_SEEN:
        return
    _ONCE_SEEN.add(msg)
    logger.info(msg, *args)

# Header: CPU-path counters at offset 0, device-path per-chunk publish
# flags at 4096 (8 ranks x 64 chunks x 8 B), device-path consumption
# counters at 8192. Slots start after the header.
_HEADER_BYTES = 65536


def _pin_host_memory(ptr: int, nbytes: int, device: torch.device) -> bool:
    """Page-lock an existing host mapping with the device's runtime.

    CUDA and ROCm expose the identical call (cudaHostRegister /
    hipHostRegister); each process only ever registers with its OWN
    runtime, which is what makes this transport vendor-neutral.
    """
    try:
        if torch.version.hip is not None:
            lib = ctypes.CDLL("libamdhip64.so")
            fn = lib.hipHostRegister
        else:
            lib = ctypes.CDLL("libcudart.so")
            fn = lib.cudaHostRegister
        fn.restype = ctypes.c_int
        # flag 0 = Default (portable mapping)
        ret = fn(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(0))
        if ret != 0:
            logger.warning(
                "HTCCL: host-register of the shm segment failed (rc=%d); "
                "falling back to unpinned copies (slower).",
                ret,
            )
        return ret == 0
    except OSError as e:
        logger.warning("HTCCL: runtime library not loadable (%s); unpinned.", e)
        return False


class HTCCLShmTransport:
    """Symmetric per-rank slots in one shm segment + seq-counter barrier."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        slot_bytes: int,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.slot_bytes = slot_bytes
        total = _HEADER_BYTES + self.world_size * slot_bytes

        # Peer identities first, BEFORE the rendezvous: the exchange is
        # itself a collective, so a rank that never shows up is caught by
        # the exchange's own deadline rather than by the unbounded
        # broadcast below. From here on every wait in this transport can
        # name the process that went away.
        self._peer_table = htccl_liveness.install(cpu_group)

        # Rendezvous: rank 0 creates the segment, broadcasts its name
        # over the (vendor-neutral) gloo cpu_group.
        if self.rank == 0:
            self._shm = shared_memory.SharedMemory(create=True, size=total)
            name = [self._shm.name]
        else:
            name = [None]
        # The object collectives run inline in torch and cannot be polled;
        # refusing to enter one whose peer is already gone is the bound.
        htccl_liveness.check_peers(
            "htccl shm rendezvous (broadcast_object_list)",
            table=self._peer_table,
        )
        dist.broadcast_object_list(name, src=dist.get_global_rank(cpu_group, 0),
                                   group=cpu_group)
        if self.rank != 0:
            self._shm = shared_memory.SharedMemory(name=name[0])

        buf = self._shm.buf
        # counters[r*8]: publish seq of rank r (uint64, cacheline padded)
        self._counters = np.frombuffer(
            buf, dtype=np.uint64, count=self.world_size * 8, offset=0
        ).reshape(self.world_size, 8)
        if self.rank == 0:
            self._counters[:] = 0
        self._slots = [
            np.frombuffer(
                buf, dtype=np.uint8, count=slot_bytes,
                offset=_HEADER_BYTES + r * slot_bytes,
            )
            for r in range(self.world_size)
        ]
        base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        self._pinned = _pin_host_memory(base, total, device)
        self._seq = 0

        # Torch views over the raw slots for zero-copy H2D/D2H.
        self._slot_tensors = [torch.from_numpy(s) for s in self._slots]

        # Everyone attached & counters zeroed before first use.
        htccl_liveness.bounded_barrier(
            cpu_group,
            "htccl shm bring-up barrier",
            table=self._peer_table,
        )
        _info_once(
            "HTCCL shm transport up: %d ranks x %d MiB slots, pinned=%s",
            self.world_size, slot_bytes // 2**20, self._pinned,
        )

    def _publish(self, phase: int) -> None:
        # x86/arm64 with CPython: aligned 8-byte store is atomic enough
        # for a monotonic counter published to spinning readers.
        self._counters[self.rank, phase] = self._seq

    def _wait_all(self, phase: int, seq: int | None = None) -> None:
        """Spin until every rank has published ``target`` for ``phase``.

        The deadline used to be a hardcoded 120 s that no peer-liveness
        information could shorten: a SIGKILLed rank was indistinguishable
        from a slow one for two minutes, and inside the JIT cold-build
        window 120 s is too SHORT for a healthy group. Both are now the
        shared module policy (``SGLANG_HTCCL_PEER_TIMEOUT_S``, scaled while
        the cold-build window is open), with the escape that a provably
        dead peer ends the wait immediately instead of at the deadline.
        """
        target = self._seq if seq is None else seq
        counters = self._counters[:, phase]

        def _ready() -> bool:
            return bool((counters >= target).all())

        try:
            htccl_liveness.bounded_poll(
                _ready,
                "htccl shm barrier",
                table=self._peer_table,
            )
        except htccl_liveness.PeerLivenessError as e:
            # Same diagnostic the hardcoded deadline used to print, now on
            # top of the reason the wait ended.
            raise type(e)(
                f"{e} HTCCL shm barrier state: target={target}, "
                f"phase={phase}, counters={counters.tolist()}."
            ) from None

    # -- pluggable-transport interface (see htccl.py "transport seam") --
    #
    # `handles` declares what this transport can serve; the communicator asks
    # instead of knowing. Encoding it here rather than at the call site is the
    # point: the slot-size ceiling is a property of the shm segment, not of
    # all_reduce.
    #
    # Deliberately UNCHANGED semantics: shm serves all_reduce only, and only
    # within slot_bytes -- exactly the condition that used to be inlined in
    # HTCCLCommunicator.all_reduce. all_gather/reduce_scatter were never
    # served by shm and still are not; making them shm-capable is a separate,
    # testable change (htccl.py notes it as the one op-coverage gap).
    HTCCL_OPS = frozenset({"all_reduce"})

    def handles(self, op: str, nbytes: int) -> bool:
        return op in self.HTCCL_OPS and nbytes <= self.slot_bytes

    def htccl_all_reduce(self, comm, inp: torch.Tensor) -> torch.Tensor:
        """This transport's all_reduce calling convention.

        shm reduces IN PLACE on a flat view, so it needs an output tensor it
        may overwrite -- it borrows the communicator's allocator for that.

        NOTE, and do not "restore" this: `_get_out_buf` returns a FRESH tensor
        per call. It once returned a stable per-(shape, dtype) buffer, which
        made two same-shape all_reduce results the SAME tensor and corrupted
        the model forward outright (garbage tokens, HTTP 200, no crash, no
        hang) on every non-device transport. `all_reduce` is documented and
        dispatched as out-of-place. See HTCCLCommunicator._get_out_buf.
        """
        out = comm._get_out_buf(inp)
        out.copy_(inp)
        self.all_reduce_(out.view(-1))
        return out

    def all_reduce_(self, flat: torch.Tensor) -> None:
        """In-place sum-all-reduce of a flat contiguous GPU tensor.

        Data path per rank: D2H into the own shm slot, publish, spin for
        peers, H2D of every peer slot, GPU-side add. Two PCIe crossings
        of payload size per peer — the same movement NCCL's SHM path
        performs, with the reduction on the GPU.
        """
        nbytes = flat.numel() * flat.element_size()
        assert nbytes <= self.slot_bytes, (
            f"HTCCL shm slot too small: {nbytes} > {self.slot_bytes}"
        )
        # Deferred consumption barrier: before overwriting the own slot,
        # make sure every rank has finished READING the previous op. In
        # steady state the peers finished long ago and this returns
        # immediately — cheaper than a hard barrier at the end of every
        # op.
        if self._seq > 0:
            self._wait_all(1, seq=self._seq)
        self._seq += 1
        my_slot = self._slot_tensors[self.rank][:nbytes].view(flat.dtype)

        # D2H: DMA straight into the (pinned) shm slot.
        my_slot.copy_(flat.view(-1), non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        self._publish(0)
        self._wait_all(0)

        # H2D of each peer slot + on-GPU accumulation.
        for r in range(self.world_size):
            if r == self.rank:
                continue
            peer = self._slot_tensors[r][:nbytes].view(flat.dtype)
            peer_dev = torch.empty_like(flat)
            peer_dev.view(-1).copy_(peer, non_blocking=True)
            flat.add_(peer_dev)
        torch.cuda.current_stream(self.device).synchronize()

        # Publish "I consumed every slot of this op"; the WAIT for the
        # peers' consumption is deferred to the start of the next op.
        self._publish(1)

    def close(self) -> None:
        try:
            self._shm.close()
            if self.rank == 0:
                self._shm.unlink()
        except Exception:
            pass
