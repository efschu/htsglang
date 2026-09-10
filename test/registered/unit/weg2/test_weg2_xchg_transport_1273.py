# SPDX-License-Identifier: Apache-2.0
"""#1273 slice S4 -- the transport, proven with no GPU.

WEG2_REUSE_SPEC_0908 section 6 / S4.  Four red-first tests are named in the
spec and are here under those names; the rest are the mutants on the DANGER
DIRECTION, which for a transport is silent wrongness: bytes that land in the
wrong place, or do not land at all, with no error anywhere.

**THE HERMETIC DOUBLE IS A FAKE DEVICE LAYER, NOT A MOCKED TRANSPORT.**
:class:`FakeDeviceOps` implements :class:`tp.DeviceOps` over file-mapped byte
arrays and a file-mediated IPC handle table.  Everything above the adapter is
the product code, unmodified: the batcher, the slot state machine, the real
POSIX semaphores, the real ``/dev/shm`` region, the short-piece refusal, the
two on-card hops, the real per-rank thread set.  Six real processes run it.  What the fake
replaces is exactly the set of calls that need silicon, and that set contains
no branch.

FOUR PROPERTIES THE FAKE KEEPS THAT A MOCK WOULD LOSE, each because a defect
of that exact shape is what this slice exists to prevent:

* **Copies are ASYNC.**  ``memcpy_async`` records the copy and does nothing;
  the bytes move inside ``synchronize``.  A consumer that read a slot before
  the producer's sync therefore reads the PREVIOUS batch, which is what makes
  ``test_consumer_never_reads_an_unposted_slot`` an ordering test and not a
  wish.
* **An imported IPC pointer is a DIFFERENT address** from the exported one, as
  S0 measured on the metal (``0x7c0f40000000`` exported, ``0x7b3180000000``
  imported).  Code that assumed the peer's address was its own would pass a
  same-address fake and fail on the rig.
* **Handles are 64 bytes with embedded NULs and a non-zero trailer.**  That is
  the ``c_char`` vs ``c_ubyte`` truncation S0 caught at the desk; a fake with
  printable handles cannot see it.
* **Device memory is per-rank and separate**, so a copy that lands on the wrong
  rank lands somewhere a checksum can find it.

EVERY DESTINATION IS PRE-POISONED, never pre-zeroed and never pre-filled with
the expected bytes: a no-op transport must be distinguishable from a correct
one BY CONSTRUCTION (R1-8, and S0's own ``dst_first_mib_zero_before_peer``
discipline).
"""

from __future__ import annotations

import ast
import ctypes
import errno
import inspect
import itertools
import mmap
import multiprocessing as mp
import os
import re
import struct
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp
from sglang.srt.weg2 import xchg_bounce as xb

#: Small enough that batching, slot alternation and the double buffer are all
#: exercised by a few kilobytes of payload.
SLOT = 4096
FAKE_DEV_BYTES = 8 << 20

#: The wave index every single-wave test runs in.  Named rather than literal
#: because ``wave`` is a STAMP the two sides compare, and a test that passed 0
#: by accident on one side and 0 by accident on the other would prove nothing
#: about the comparison.  ``test_a_previous_waves_oncard_row_is_not_this_waves_
#: signal`` is the one that moves it.
WAVE = 0

_COUNTER = itertools.count()


def _fresh_boot() -> str:
    """A boot nonce unique to this process AND this test.

    The 24 semaphore names are GLOBAL to the machine and
    ``create_semaphores`` unlinks before creating, so a fixed nonce would make
    two tests running under ``-n 24`` on the remote desk silently destroy each
    other's handshake -- and the symptom would be a producer blocking to the
    fence budget, i.e. a flake that reads exactly like the defect this slice
    is about.
    """
    return f"s4b{os.getpid()}x{next(_COUNTER)}"


# ===========================================================================
# The fake device layer.
# ===========================================================================

#: A TAG BIT, above the 47-bit x86-64 user address range, not a base address to
#: compare against.  MEASURED DEFECT, first remote run of this file: with the
#: tag at ``1 << 46`` and the split written as ``ptr < FAKE_DEV_BASE``, every
#: REAL host address -- the mapped region's ``0x7f...``, i.e. ~1.4e14 -- sorted
#: as a DEVICE pointer, was decoded into rank 1 of a file that had never been
#: sized for it, and the first staging copy segfaulted.  A harness that
#: mis-sorts pointers cannot prove anything about a transport that moves them,
#: so the split is a tag test now and is pinned by a test of its own.
FAKE_DEV_TAG = 1 << 50
FAKE_RANK_SHIFT = 40
FAKE_IMPORT_BIAS = 1 << 49


def dev_ptr(rank: int, off: int) -> int:
    return FAKE_DEV_TAG | (int(rank) << FAKE_RANK_SHIFT) | int(off)


class FakeDeviceOps(tp.DeviceOps):
    """:class:`tp.DeviceOps` over bytes.  Cross-process by construction.

    "Device memory" is one file per rank under ``root/dev-<rank>.bin``; a
    device pointer is ``FAKE_DEV_TAG | rank << 40 | offset``, so which rank a
    copy landed on is recoverable from the address alone.  Host pointers are
    REAL addresses (the mapped shm region), so the staging layout under test is
    the product's own and not a model of it -- which is exactly why the
    host/device split must be a TAG test and not a magnitude one.
    """

    name = "fake"

    def __init__(self, root: str, rank: int, *, fail_ipc_open: bool = False,
                 fail_host_register: bool = False, drop_last_copy: bool = False,
                 stream_flags: int = tp.CUDA_STREAM_DEFAULT,
                 on_free=None):
        self.root = root
        self.rank = int(rank)
        self.fail_ipc_open = fail_ipc_open
        self.fail_host_register = fail_host_register
        self.drop_last_copy = drop_last_copy
        self.on_free = on_free
        self._stream_flags = int(stream_flags)
        self._maps: dict = {}
        self._bump = 0
        self._streams: dict = {}
        self._next_stream = 1
        self._lock = threading.Lock()
        self.registered: list = []
        self.issued = 0
        self.freed: list = []
        self.closed_handles: list = []
        self._allocs: dict = {}
        os.makedirs(os.path.join(root, "ipc"), exist_ok=True)

    # -- storage ----------------------------------------------------------
    def _base_of(self, rank: int) -> int:
        with self._lock:
            got = self._maps.get(rank)
            if got is not None:
                return got[0]
            path = os.path.join(self.root, f"dev-{rank}.bin")
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            if os.fstat(fd).st_size < FAKE_DEV_BYTES:
                os.ftruncate(fd, FAKE_DEV_BYTES)
            mm = mmap.mmap(fd, FAKE_DEV_BYTES, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
            holder = ctypes.c_char.from_buffer(mm)
            base = ctypes.addressof(holder)
            del holder
            self._maps[rank] = (base, mm, fd)
            return base

    @staticmethod
    def decode(ptr: int) -> tuple:
        """``(rank, offset)`` of a device pointer, or ``None`` for a host one.

        A TAG TEST, never a magnitude comparison -- see :data:`FAKE_DEV_TAG`.
        """
        raw = int(ptr) & ~FAKE_IMPORT_BIAS
        if not raw & FAKE_DEV_TAG:
            return None
        body = raw & ~FAKE_DEV_TAG
        return (body >> FAKE_RANK_SHIFT, body & ((1 << FAKE_RANK_SHIFT) - 1))

    def real(self, ptr: int) -> int:
        """Resolve a fake pointer to a real address.  Host pointers pass through."""
        got = self.decode(ptr)
        if got is None:
            return int(ptr)
        rank, off = got
        assert off + 0 <= FAKE_DEV_BYTES, (rank, off)
        return self._base_of(rank) + off

    # -- streams ----------------------------------------------------------
    def create_stream(self, device: int) -> int:
        with self._lock:
            stream = self._next_stream
            self._next_stream += 1
        self._streams[stream] = []
        return stream

    def stream_flags(self, stream: int) -> int:
        return self._stream_flags

    def destroy_stream(self, stream: int) -> None:
        self._streams.pop(stream, None)

    def pending_total(self) -> int:
        """How many issued copies have not executed yet.

        The fake defers every copy to ``synchronize``, so this is the
        honest reading of 'bytes issued but not landed' -- the quantity
        the whole ``bytes_filled`` handshake exists to keep out of a
        consumer's hands.
        """
        return sum(len(q) for q in self._streams.values())

    def synchronize(self, stream: int) -> None:
        queued = self._streams.get(stream, [])
        if self.drop_last_copy and queued:
            queued = queued[:-1]
        for fn in queued:
            fn()
        self._streams[stream] = []

    # -- copies -----------------------------------------------------------
    def memcpy_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        d, s, n = self.real(dst), self.real(src), int(nbytes)
        self.issued += 1
        self._streams[stream].append(lambda: ctypes.memmove(d, s, n))

    def memcpy2d_async(self, dst: int, dpitch: int, src: int, spitch: int,
                       width: int, height: int, stream: int) -> None:
        d, s = self.real(dst), self.real(src)
        self.issued += 1

        def run() -> None:
            for row in range(int(height)):
                ctypes.memmove(d + row * int(dpitch), s + row * int(spitch),
                               int(width))

        self._streams[stream].append(run)

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        d, n, v = self.real(dst), int(nbytes), int(value)
        self.issued += 1
        self._streams[stream].append(lambda: ctypes.memset(d, v, n))

    # -- host pinning -----------------------------------------------------
    def host_register(self, ptr: int, nbytes: int, flags: int) -> None:
        if self.fail_host_register:
            raise RuntimeError("cudaHostRegister rc=2 out of memory")
        self.registered.append((int(ptr), int(nbytes), int(flags)))

    def host_unregister(self, ptr: int) -> None:
        self.registered = [r for r in self.registered if r[0] != int(ptr)]

    # -- the on-card lane -------------------------------------------------
    def raw_malloc(self, device: int, nbytes: int) -> int:
        with self._lock:
            off = self._bump
            self._bump += int(nbytes)
        assert self._bump <= FAKE_DEV_BYTES, "fake device out of memory"
        self._base_of(self.rank)
        ptr = dev_ptr(self.rank, off)
        self._allocs[ptr] = int(nbytes)
        return ptr

    def raw_free(self, ptr: int) -> None:
        """A FREE THAT POISONS, because a free that does nothing cannot be
        used to prove anything about a lifetime.

        ``cudaFree`` returns the pages to the driver, which may hand them to
        anyone; a peer still copying out of an imported mapping then reads
        whatever is there.  The first version of this fake returned ``None``,
        which is exactly why 36 green tests and six mutants could not see the
        producer freeing its exported bounce while its co-located consumer was
        still reading it (S4 review + refuter, must_fix).  Poisoning makes that
        window a byte mismatch instead of a shrug.
        """
        nbytes = self._allocs.pop(int(ptr), 0)
        self.freed.append((int(ptr), nbytes))
        if nbytes:
            ctypes.memset(self.real(int(ptr)), 0x6B, nbytes)
        if self.on_free is not None:
            self.on_free(int(ptr), nbytes)

    def ipc_get_handle(self, ptr: int) -> bytes:
        got = self.decode(ptr)
        assert got is not None, "an IPC handle is only ever taken on device memory"
        rank, off = got
        token = f"{rank}-{off}".encode("ascii")
        with open(os.path.join(self.root, "ipc", f"h{rank}-{off}"), "wb") as fh:
            fh.write(struct.pack("<QQ", rank, off))
        # 64 bytes: the token, then NULs, then a NON-ZERO trailer.  A c_char
        # round trip truncates at the first NUL and loses the trailer -- which
        # is exactly the defect S0 caught before it reached the metal.
        blob = token + b"\x00" * (tp.CUDA_IPC_HANDLE_SIZE - len(token) - 1) + b"\xa5"
        assert len(blob) == tp.CUDA_IPC_HANDLE_SIZE
        return blob

    def ipc_open_handle(self, handle: bytes) -> int:
        if self.fail_ipc_open:
            raise RuntimeError("cudaIpcOpenMemHandle rc=1 invalid argument")
        if len(handle) != tp.CUDA_IPC_HANDLE_SIZE:
            raise ValueError(f"short handle: {len(handle)} bytes")
        if handle[-1] != 0xA5:
            raise ValueError(
                "the handle's trailer is gone -- this is the c_char truncation")
        token = handle.split(b"\x00", 1)[0].decode("ascii")
        rank_s, off_s = token.split("-")
        rank, off = int(rank_s), int(off_s)
        self._base_of(rank)
        # A DIFFERENT address than the exporter's, as S0 measured.
        return dev_ptr(rank, off) | FAKE_IMPORT_BIAS

    def ipc_close_handle(self, ptr: int) -> None:
        self.closed_handles.append(int(ptr))

    def close(self) -> None:
        for _base, mm, fd in self._maps.values():
            mm.close()
            os.close(fd)
        self._maps.clear()


# ===========================================================================
# Fixtures and helpers.
# ===========================================================================


def poison(ops: FakeDeviceOps, ptr: int, nbytes: int, seed: int) -> None:
    """Fill a range with a recognisable non-zero pattern.

    NEVER with the expected bytes and never with zero: the destination has to
    be distinguishable from both a no-op and a correct copy, or the assertion
    that follows it can pass for the wrong reason.
    """
    ctypes.memmove(ops.real(ptr),
                   bytes(((seed + i * 3) & 0xFF) for i in range(nbytes)), nbytes)


def pattern(seed: int, nbytes: int) -> bytes:
    return bytes(((seed * 7 + i * 13) & 0xFF) for i in range(nbytes))


def read(ops: FakeDeviceOps, ptr: int, nbytes: int) -> bytes:
    return ctypes.string_at(ops.real(ptr), nbytes)


def write(ops: FakeDeviceOps, ptr: int, payload: bytes) -> None:
    ctypes.memmove(ops.real(ptr), payload, len(payload))


def flat_desc(src_rank, dst_rank, nbytes, *, src_off=0, dst_off=0,
              src_ptr=None, dst_ptr=None, name="flat", tag="weights_0"):
    return wx.XchgDesc(
        tag=tag, src_rank=src_rank, dst_rank=dst_rank, param_name=name,
        kind=tp.FLAT, nbytes=nbytes, rows=1, run_bytes=nbytes,
        spitch=0, dpitch=0, src_off=src_off, dst_off=dst_off,
        src_ptr=src_ptr, dst_ptr=dst_ptr,
    )


def strided_desc(src_rank, dst_rank, *, rows, run_bytes, spitch, dpitch,
                 src_off=0, dst_off=0, src_ptr=None, dst_ptr=None,
                 name="strided", tag="weights_0"):
    return wx.XchgDesc(
        tag=tag, src_rank=src_rank, dst_rank=dst_rank, param_name=name,
        kind=tp.STRIDED2D, nbytes=rows * run_bytes, rows=rows,
        run_bytes=run_bytes, spitch=spitch, dpitch=dpitch,
        src_off=src_off, dst_off=dst_off, src_ptr=src_ptr, dst_ptr=dst_ptr,
    )


@pytest.fixture()
def boot():
    return _fresh_boot()


@pytest.fixture()
def region(tmp_path, boot):
    r = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    r.begin_flip(f"{boot}.1")
    yield r
    r.close()


@pytest.fixture()
def sems(boot):
    xr.create_semaphores(boot)
    s = tp.SemSet(boot)
    yield s
    s.close()
    xr.unlink_semaphores(boot)


@pytest.fixture()
def ops(tmp_path):
    o = FakeDeviceOps(str(tmp_path / "dev"), rank=0)
    yield o
    o.close()


def _round_trip(region, sems, ops, descs, *, pair, slot_bytes=SLOT,
                budget_s=10.0):
    """Run one pair's producer and consumer CONCURRENTLY, in two threads.

    Concurrently and not one after the other, because the double buffer is only
    two slots deep: a serial run of more than two batches blocks the producer on
    ``empty`` forever.  A helper that quietly worked only for short payloads
    would put a ceiling on every test in this file without saying so.
    """
    pstats = tp.PairStats(*xr.CROSS_PAIRS[pair], "us", "ud")
    cstats = tp.PairStats(*xr.CROSS_PAIRS[pair], "us", "ud")
    errors: list = []

    def produce():
        try:
            tp.run_producer_pair(region, sems, ops, ops.create_stream(0),
                                 pair=pair, descs=descs, stats=pstats,
                                 budget_s=budget_s, slot_bytes=slot_bytes)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        tp.run_consumer_pair(region, sems, ops, ops.create_stream(0),
                             pair=pair, descs=descs, stats=cstats,
                             budget_s=budget_s, slot_bytes=slot_bytes)
    finally:
        thread.join(30)
    assert not errors, errors
    return pstats, cstats


# ===========================================================================
# THE FOUR RED-FIRST TESTS NAMED BY SPEC SECTION 6 / S4.
# ===========================================================================


def test_short_piece_is_refused_by_the_consumer(region, sems, ops):
    """The producer posts ``bytes_filled = plan - 1``; the consumer raises W70
    and issues NO copy.

    This is #802's physical root, which #802 itself explicitly did not close.
    The two halves of the assertion carry equal weight: the NAME must be in the
    message so a grep over a boot log finds it, and ``ops.issued`` must not
    move, so the refusal is PROVEN to precede the copy rather than to follow
    it.  A W70 raised after the copy is a log line, not a guard.
    """
    pair = xr.pair_id(0, 1)
    payload = pattern(3, 600)
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    write(ops, src, payload)
    poison(ops, dst, len(payload), seed=0x11)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst)]
    batch = tp.batch_descs(descs, SLOT)[0]

    # The producer, by hand, one byte short of what the plan claims.
    assert sems.trywait(pair, batch.slot, "empty")
    region.begin_fill(pair, batch.slot, batch.seq)
    stream = ops.create_stream(0)
    ops.memcpy_async(region.slot_address(pair, batch.slot), src, len(payload),
                     stream)
    ops.synchronize(stream)
    region.publish(pair, batch.slot, batch.total_bytes - 1)
    sems.post(pair, batch.slot, "full")

    before = ops.issued
    with pytest.raises(tp.Weg2XchgShortPiece) as excinfo:
        tp.run_consumer_pair(region, sems, ops, ops.create_stream(0),
                             pair=pair, descs=descs,
                             stats=tp.PairStats(0, 1, "u0", "u1"),
                             budget_s=2.0, slot_bytes=SLOT)
    message = str(excinfo.value)
    assert tp.SHORT_PIECE_MARKER in message
    assert f"expected_bytes={batch.total_bytes}" in message
    assert f"bytes_filled={batch.total_bytes - 1}" in message
    assert "short_by=1" in message
    assert "lane=cross" in message
    assert ops.issued == before, "W70 must precede the first copy, not follow it"
    assert read(ops, dst, len(payload)) != payload


def test_bytes_filled_is_published_only_after_the_sync(region, sems, ops):
    """``bytes_filled`` must mean "these bytes have LANDED", not "were issued".

    ADDED AFTER A SURVIVING MUTANT.  Swapping ``synchronize`` and ``publish``
    in :func:`tp.run_producer_pair` killed NOTHING in the first mutant round:
    every test reached the slot through ``sem_post(full)``, which still came
    after the sync, so the suite was proving ``sync -> post`` and had never
    looked at ``sync -> publish``.  That is #802 rule 2 exactly -- the number
    the consumer copies must be a post-sync fact -- and S5's shadow will read
    the slot record's ``checksum`` field without the semaphore at all, at which
    point the untested half becomes the load-bearing one.

    The assertion is behavioural, not structural: at the instant ``publish`` is
    called, the fake must hold ZERO issued-but-unexecuted copies.
    """
    pair = xr.pair_id(0, 1)
    payload = pattern(41, 6000)          # 2 batches at SLOT, so no consumer
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    write(ops, src, payload)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst)]
    assert len(tp.batch_descs(descs, SLOT)) == 2

    pending_at_publish: list = []
    real_publish = region.publish

    def spy(pair_, slot_, nbytes, **kw):
        pending_at_publish.append(ops.pending_total())
        return real_publish(pair_, slot_, nbytes, **kw)

    region.publish = spy
    try:
        tp.run_producer_pair(region, sems, ops, ops.create_stream(0), pair=pair,
                             descs=descs, stats=tp.PairStats(0, 1, "a", "b"),
                             budget_s=2.0, slot_bytes=SLOT)
    finally:
        del region.publish
    assert len(pending_at_publish) == 2
    assert pending_at_publish == [0, 0], (
        "publish ran with copies still unexecuted -- bytes_filled would then "
        f"mean 'issued', not 'landed': {pending_at_publish}")

    # Independent second guard, in case a future edit moves the sync into a
    # helper the fake cannot see: the three statements in that order, by AST.
    body = ast.parse(inspect.getsource(tp.run_producer_pair).lstrip()).body[0]
    order = []
    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name in ("synchronize", "publish", "post"):
            order.append((node.lineno, name))
    order = [n for _l, n in sorted(order)]
    assert order[-3:] == ["synchronize", "publish", "post"], order


def test_consumer_never_reads_an_unposted_slot(region, sems, ops):
    """With ``full`` unposted the consumer waits its budget and then names the
    wait -- it does not read the slot.

    The slot is deliberately pre-filled with the RIGHT bytes before the test: a
    consumer that read it without the post would produce a correct destination
    and look like a pass.  The failure this guards is not "wrong bytes", it is
    "right bytes for the wrong reason" -- which on a real flip is the PREVIOUS
    batch's payload sitting in the same slot.
    """
    pair = xr.pair_id(0, 1)
    payload = pattern(5, 800)
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    write(ops, src, payload)
    poison(ops, dst, len(payload), seed=0x22)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst)]
    batch = tp.batch_descs(descs, SLOT)[0]
    ctypes.memmove(region.slot_address(pair, batch.slot), payload, len(payload))
    region.begin_fill(pair, batch.slot, batch.seq)
    region.publish(pair, batch.slot, batch.total_bytes)

    before = ops.issued
    stats = tp.PairStats(0, 1, "u0", "u1")
    started = time.monotonic()
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp.run_consumer_pair(region, sems, ops, ops.create_stream(0),
                             pair=pair, descs=descs, stats=stats,
                             budget_s=0.25, slot_bytes=SLOT)
    assert "W69 Weg2XchgGateTimeout" in str(excinfo.value)
    assert "kind=full" in str(excinfo.value)
    assert time.monotonic() - started >= 0.2, "it must actually have waited"
    assert ops.issued == before
    assert read(ops, dst, len(payload)) != payload
    assert stats.slot_waits == 1 and stats.slot_wait_s > 0


class _StubCudart:
    """The narrowest possible stand-in for :class:`tp.CudartDeviceOps`' state.

    It exists so ``create_stream``'s flag check -- the one guard that lives in
    the CUDA-only class and therefore cannot be reached through the fake -- can
    be executed on a desk.  Anything more here would be testing the stub.
    """

    def __init__(self, flags: int):
        self._flags = flags
        self.destroyed: list = []

    def set_device(self, device):
        return None

    @property
    def lib(self):
        class L:
            @staticmethod
            def cudaStreamCreate(ref):
                ref._obj.value = 0x1234
                return 0

        return L

    def _check(self, rc, what):
        assert rc == 0, what

    def stream_flags(self, stream):
        return self._flags

    def destroy_stream(self, stream):
        self.destroyed.append(stream)


def test_stream_is_not_nonblocking():
    """Three independent guards on R12, because one of them is only a comment.

    (a) the real adapter reads the flags BACK and refuses a non-zero value --
        "we called the blocking constructor" is exactly the kind of claim that
        survives a refactor while ceasing to be true;
    (b) the refusal names the HAZARD, not the flag, so a reader learns why;
    (c) no non-blocking token appears anywhere in the module's CODE (an AST
        walk, so the module's own prose about the rule cannot mask a call).
    """
    stub = _StubCudart(flags=1)
    with pytest.raises(RuntimeError) as excinfo:
        tp.CudartDeviceOps.create_stream(stub, 0)
    assert "cudaStreamDefault" in str(excinfo.value)
    assert "pending kernel" in str(excinfo.value)
    assert stub.destroyed == [0x1234], "a refused stream must still be destroyed"

    good = _StubCudart(flags=tp.CUDA_STREAM_DEFAULT)
    assert tp.CudartDeviceOps.create_stream(good, 0) == 0x1234

    tree = ast.parse(inspect.getsource(tp))
    banned = {"cudaStreamCreateWithFlags", "cudaStreamNonBlocking"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            hits.append((node.attr, node.lineno))
        if isinstance(node, ast.Name) and node.id in banned:
            hits.append((node.id, node.lineno))
    assert hits == [], f"non-blocking stream token on a CODE line: {hits}"


def test_oncard_lane_falls_back_by_name(tmp_path):
    """A failing ``cudaIpcOpenMemHandle`` arms **W72** at LAUNCH and the lane
    degrades with a named line that says WHERE TO.

    MEASURED-BY-REVIEW DEFECT (S4 review, 2026-09-09, must_fix): the line said
    "the on-card share routes through the staging region", which is what spec
    3.7 degrade 3 and spec 6/S4 both say and is NOT what the code does -- the
    degrade uses a per-card bounce FILE (``oncard_host_path``), because
    ``CROSS_PAIRS`` has no diagonal to borrow.  The old assertions checked the
    code, the cost and the arming discipline and never the TARGET, so they
    passed straight over it; a reader of the line would have concluded the
    degrade was ledger-neutral when it adds host bytes spec 0.2 does not
    carry.  The target is asserted here against the function that actually
    builds the path, so the two cannot drift apart again.

    Five things now: the code; the degrade TARGET and its host term; that the
    decision is taken at the ARM (``probe`` runs exactly once, so nothing
    per-flip and nothing per-lane can change it later -- R2-1); that an
    explicit request for the degrade is not overridden by a green probe; and
    that an explicit request for ``ipc`` is REFUSED rather than silently
    degraded.
    """
    calls: list = []
    lines: list = []

    def probe():
        calls.append(1)
        ops = FakeDeviceOps(str(tmp_path / "p"), rank=0, fail_ipc_open=True)
        try:
            ops.ipc_open_handle(ops.ipc_get_handle(ops.raw_malloc(0, 4096)))
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            ops.close()

    assert tp.arm_oncard_lane(card_uuid="GPU-abc", probe=probe,
                              log=lines.append) == tp.ONCARD_MODE_HOST
    assert len(calls) == 1
    assert len(lines) == 1
    assert tp.ONCARD_UNAVAILABLE_MARKER in lines[0]
    assert "card=GPU-abc" in lines[0]
    assert f"degrade_to={tp.ONCARD_MODE_HOST}" in lines[0]
    assert "cudaIpcOpenMemHandle" in lines[0]
    assert "0.2" in lines[0] and "host saving" in lines[0], (
        "the degrade must state its cost and what it does NOT cost")
    # THE TARGET, tied to the function that builds it rather than to prose.
    bounce_name = os.path.basename(tp.oncard_host_path("bootx", 7))
    assert bounce_name == "oncard-7.bin", bounce_name
    assert f"bounce={bounce_name.replace('7', '<card>')}" in lines[0], lines[0]
    assert "NOT" in lines[0] and "staging region" in lines[0], (
        "the line must say which target it does NOT use, or the spec wording "
        "keeps reading as the behaviour")
    assert f"host_add_mib={tp.ONCARD_HOST_DEGRADE_MIB}" in lines[0]
    # ...and the term is the real geometry: one bounce per card of the three,
    # ONCARD_SLOTS slots each.  Computed here from the same two constants the
    # HostBounce is built from, never from the message.
    # S6 must_fix 3: the CEILING of the shape, not one plan's choice.  The
    # line is printed at ARM time, before any plan has derived a slot size.
    assert tp.ONCARD_HOST_DEGRADE_MIB == \
        3 * tp.ONCARD_DEPOSIT_BYTES_MAX // xr.MIB

    ok_lines: list = []
    assert tp.arm_oncard_lane(card_uuid="GPU-abc", probe=lambda: (True, ""),
                              log=ok_lines.append) == tp.ONCARD_MODE_IPC
    assert ok_lines == []

    never: list = []
    req_lines: list = []
    assert tp.arm_oncard_lane(
        card_uuid="GPU-abc",
        probe=lambda: (never.append(1), (True, ""))[1],
        requested=tp.ONCARD_MODE_HOST, log=req_lines.append,
    ) == tp.ONCARD_MODE_HOST
    assert never == [], "a green probe must not override an explicit --oncard host"
    assert tp.ONCARD_UNAVAILABLE_MARKER in req_lines[0]

    # A probe that RAISES is a NO, not a crash at the launcher.
    raised: list = []
    assert tp.arm_oncard_lane(
        card_uuid="GPU-abc",
        probe=lambda: (_ for _ in ()).throw(OSError("no such device")),
        log=raised.append) == tp.ONCARD_MODE_HOST
    assert "OSError" in raised[0]

    # AN EXPLICIT `ipc` WHOSE PROBE FAILS IS A REFUSAL, and it is the only
    # runtime use the W72 CLASS has.  Degrading here would run a boot whose
    # operator believes the lane is on, and every wall claim that followed
    # would be about a lane that is not there.  A refusal that is only ever a
    # string is a refusal nobody can catch.
    asked: list = []
    with pytest.raises(tp.Weg2XchgOnCardUnavailable) as excinfo:
        tp.arm_oncard_lane(card_uuid="GPU-abc",
                           probe=lambda: (False, "cudaIpc rc=1"),
                           requested=tp.ONCARD_MODE_IPC, log=asked.append)
    assert tp.ONCARD_UNAVAILABLE_MARKER in str(excinfo.value)
    assert "degrade_to=none" in str(excinfo.value)
    assert f"requested={tp.ONCARD_MODE_IPC}" in str(excinfo.value)
    assert asked and tp.ONCARD_UNAVAILABLE_MARKER in asked[0], (
        "a refusal must also reach the log: the raise unwinds into the "
        "launcher, and the boot log is where the reason has to survive")


def test_the_degraded_lane_really_moves_the_bytes(tmp_path, region):
    """W72 names a degrade; the degrade has to work.

    A named fallback that was never executed is the
    desk-written-never-executed class.  The ``host`` arm runs the same batcher,
    the same rows and the same W70 comparison over a per-card shm bounce
    instead of an IPC one.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    cons = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(31, 6000)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(prod, src, payload)
        poison(cons, dst, len(payload), seed=0xEE)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]
        p_bounce = tp.HostBounce(prod, region.boot_nonce, 0, create=True,
                                 slot_bytes=SLOT, shm_root=os.path.dirname(
                                     os.path.dirname(region.path)))
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                            slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_ARMED)
        c_bounce = tp.HostBounce(cons, region.boot_nonce, 0, create=False,
                                 slot_bytes=SLOT, shm_root=os.path.dirname(
                                     os.path.dirname(region.path)))
        assert c_bounce.ptr != p_bounce.ptr, "two mappings, two addresses"
        # The degrade's host cost is a REAL file of a size this test can read,
        # and it scales exactly as the W72 line's `host_add_mib` claims.
        assert os.path.getsize(p_bounce.path) == tp.ONCARD_SLOTS * SLOT
        assert p_bounce.path.endswith("oncard-0.bin"), p_bounce.path
        assert tp.ONCARD_HOST_DEGRADE_MIB * xr.MIB == \
            3 * tp.ONCARD_DEPOSIT_BYTES_MAX
        errors: list = []

        def produce():
            try:
                tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                       p_bounce, row=0, peer_row=3, wave=WAVE,
                                       descs=descs,
                                       stats=tp.OnCardStats(0, "u0", "host"),
                                       budget_s=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_HOST)
        tp.run_oncard_consumer(region, cons, cons.create_stream(0), c_bounce.ptr,
                               row=3, peer_row=0, wave=WAVE, descs=descs,
                               stats=stats, slot_bytes=SLOT, budget_s=10.0)
        thread.join(30)
        assert not errors, errors
        assert read(cons, dst, len(payload)) == payload
        assert "mode=host" in stats.line()
        p_bounce.close()
        c_bounce.close()
    finally:
        prod.close()
        cons.close()


# ===========================================================================
# MUTANTS ON THE DANGER DIRECTION: byte identity per class.
# ===========================================================================


def test_flat_and_fused_subblocks_land_byte_identical(region, sems, ops):
    """A fused parameter's FOUR device sub-blocks each land at their own offset.

    ``in_proj_qkvz`` is one buffer of four rank-local sub-blocks
    (``output_sizes=[2048,2048,6144,6144]``); a plan built from the checkpoint's
    three offsets writes ``k`` over ``q`` on ranks 1 and 2 -- plausible garbage,
    no error (R5).  The transport cannot detect a wrong offset; what it must
    guarantee is that it REPRODUCES the offsets it is given, exactly, including
    where a sub-block straddles a slot boundary.  The mutant proves the
    assertion can fail.
    """
    pair = xr.pair_id(0, 1)
    blocks = [(0, 900), (900, 900), (1800, 2700), (4500, 2700)]  # q k v z
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    total = blocks[-1][0] + blocks[-1][1]
    expect = bytearray(total)
    for index, (off, size) in enumerate(blocks):
        blob = pattern(index + 1, size)
        write(ops, src + off, blob)
        expect[off:off + size] = blob
    poison(ops, dst, total, seed=0x33)

    descs = [flat_desc(0, 1, size, src_off=off, dst_off=off,
                       src_ptr=src, dst_ptr=dst, name=f"qkvz.{i}")
             for i, (off, size) in enumerate(blocks)]
    assert len(tp.batch_descs(descs, SLOT)) > 1, "no slot boundary was crossed"
    _round_trip(region, sems, ops, descs, pair=pair)
    assert read(ops, dst, total) == bytes(expect)

    # MUTANT: the third sub-block's destination offset shifted onto the second.
    poison(ops, dst, total, seed=0x44)
    mutated = list(descs)
    mutated[2] = mutated[2].replace(dst_off=blocks[1][0])
    _round_trip(region, sems, ops, mutated, pair=pair)
    assert read(ops, dst, total) != bytes(expect), (
        "a wrong device sub-block offset must not produce an identical "
        "destination -- if it does, this test proves nothing")


def test_strided_2d_lands_byte_identical_and_touches_no_padding(region, sems, ops):
    """A row-parallel class compacts into the slot and scatters out of it.

    ``mlp down_proj``'s shape: 5120 runs at one pitch in the source's arena and
    a wider pitch in the destination's.  Only payload crosses; the
    destination's padding columns must be left EXACTLY as they were, because
    they belong to the other ranks' shards -- a scatter that widened by one
    column would corrupt a neighbour's weights with no error anywhere.
    """
    pair = xr.pair_id(0, 1)
    rows, run, spitch, dpitch, dcol = 12, 96, 128, 320, 64
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    src_bytes, dst_bytes = rows * spitch, rows * dpitch
    write(ops, src, pattern(9, src_bytes))
    poison(ops, dst, dst_bytes, seed=0x55)
    before = read(ops, dst, dst_bytes)

    desc = strided_desc(0, 1, rows=rows, run_bytes=run, spitch=spitch,
                        dpitch=dpitch, dst_off=dcol, src_ptr=src, dst_ptr=dst,
                        name="down_proj")
    _round_trip(region, sems, ops, [desc], pair=pair)

    got = read(ops, dst, dst_bytes)
    src_now = read(ops, src, src_bytes)
    for row in range(rows):
        base = row * dpitch
        assert got[base + dcol: base + dcol + run] == \
            src_now[row * spitch: row * spitch + run], f"row {row} payload"
        assert got[base: base + dcol] == before[base: base + dcol], \
            f"row {row} left padding"
        assert got[base + dcol + run: base + dpitch] == \
            before[base + dcol + run: base + dpitch], f"row {row} right padding"

    # MUTANT 1: the two pitches swapped -- the classic 2-D transposition.
    poison(ops, dst, dst_bytes, seed=0x66)
    _round_trip(region, sems, ops,
                [desc.replace(spitch=dpitch, dpitch=spitch)], pair=pair)
    assert read(ops, dst, dst_bytes) != got

    # MUTANT 2: the run read as the source pitch -- what a FLAT reading of this
    # class produces, i.e. padding copied as payload.
    poison(ops, dst, dst_bytes, seed=0x77)
    _round_trip(region, sems, ops, [desc.replace(spitch=run)], pair=pair)
    assert read(ops, dst, dst_bytes) != got

    # MUTANT 3: one row short.  A silent truncation at the tail is the shape a
    # length assertion over the whole buffer would miss.
    poison(ops, dst, dst_bytes, seed=0x78)
    _round_trip(region, sems, ops, [desc.replace(rows=rows - 1,
                                                 nbytes=(rows - 1) * run)],
                pair=pair)
    assert read(ops, dst, dst_bytes) != got


def test_a_no_op_transport_cannot_pass(region, sems, ops):
    """A dropped copy must be visible.

    The fake's ``drop_last_copy`` removes the final queued copy at the sync --
    a producer whose last piece never lands, i.e. the exact shape of a
    swallowed ``cudaMemcpyAsync`` return code.  Without this the whole file
    could be passing on pre-arranged bytes.
    """
    pair = xr.pair_id(0, 1)
    payload = pattern(11, 1500)
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    write(ops, src, payload)
    poison(ops, dst, len(payload), seed=0x88)
    descs = [flat_desc(0, 1, 700, src_ptr=src, dst_ptr=dst, name="a"),
             flat_desc(0, 1, 800, src_off=700, dst_off=700,
                       src_ptr=src, dst_ptr=dst, name="b")]
    _round_trip(region, sems, ops, descs, pair=pair)
    assert read(ops, dst, len(payload)) == payload

    poison(ops, dst, len(payload), seed=0x99)
    ops.drop_last_copy = True
    try:
        _round_trip(region, sems, ops, descs, pair=pair)
    finally:
        ops.drop_last_copy = False
    assert read(ops, dst, len(payload)) != payload


def test_zerofill_is_applied_and_never_staged(ops):
    """The 128 vocab pad rows are a memset on the destination, not a transfer.

    They have no VRAM source and no checkpoint source, so a batcher that let
    them into a slot would dereference ``src_ptr=None``; a design that left
    them out entirely would ship undefined pad rows.  Both halves are asserted,
    and so is the third: a rank that does not own the range must not touch it.
    """
    dst = dev_ptr(1, 0)
    nbytes = 512
    poison(ops, dst, nbytes, seed=0xAA)
    desc = wx.XchgDesc(
        tag="weights", src_rank=-1, dst_rank=1, param_name="embed_tokens",
        kind=tp.ZEROFILL, nbytes=nbytes, rows=1, run_bytes=nbytes,
        spitch=0, dpitch=0, dst_ptr=dst,
    )
    assert tp.batch_descs([desc], SLOT) == []
    assert tp.apply_zerofill(ops, ops.create_stream(0), [desc], 1) == nbytes
    assert read(ops, dst, nbytes) == b"\x00" * nbytes

    poison(ops, dst, nbytes, seed=0xBB)
    assert tp.apply_zerofill(ops, ops.create_stream(0), [desc], 0) == 0
    assert read(ops, dst, nbytes) != b"\x00" * nbytes


def test_batches_are_the_same_whether_the_side_coalesced_or_not():
    """S1's coalescer is ASYMMETRIC; the batch boundaries must not be.

    ``build_plan`` computes ``plan_id`` from the RAW list precisely because
    "which pieces merge depends on the pointer table, and the two processes
    hold different tables".  So the producer may hold one merged descriptor
    where the consumer holds its two halves.  The transport packs by RUNNING
    BYTE COUNT over descriptors that are a partition of the same ordered byte
    stream, so both must yield identical sequence numbers, identical
    ``total_bytes``, and the same bytes at the same slot offsets.

    If this were false, EVERY boot would fire W70 and the refusal would be
    about the plan builder rather than about a defect -- which is the worst
    kind of guard, one that cries wolf on the healthy path.
    """
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    halves = [flat_desc(0, 1, 3000, src_ptr=src, dst_ptr=dst, name="p"),
              flat_desc(0, 1, 4000, src_off=3000, dst_off=3000,
                        src_ptr=src, dst_ptr=dst, name="p")]
    merged = [flat_desc(0, 1, 7000, src_ptr=src, dst_ptr=dst, name="p")]
    a, b = tp.batch_descs(halves, SLOT), tp.batch_descs(merged, SLOT)
    assert [x.seq for x in a] == [x.seq for x in b]
    assert [x.total_bytes for x in a] == [x.total_bytes for x in b]

    def stream(batches):
        """(seq, slot_off, src_off, dst_off) per BYTE, folded back into runs."""
        out = []
        for batch in batches:
            for piece in batch.pieces:
                row = (batch.seq, piece.slot_off, piece.nbytes, piece.src_off,
                       piece.dst_off)
                if out and out[-1][0] == row[0] \
                        and out[-1][1] + out[-1][2] == row[1] \
                        and out[-1][3] + out[-1][2] == row[3] \
                        and out[-1][4] + out[-1][2] == row[4]:
                    seq, so, n, sf, df = out[-1]
                    out[-1] = (seq, so, n + row[2], sf, df)
                else:
                    out.append(row)
        return out

    assert stream(a) == stream(b)
    # And the merged side really did issue fewer copies, or the equality above
    # would be vacuous.
    assert sum(len(x.pieces) for x in b) < sum(len(x.pieces) for x in a)


def test_a_run_that_cannot_fit_a_slot_is_refused_before_any_copy():
    """A 2-D run wider than a slot has no staging shape, and saying so is not
    optional: silently splitting a run mid-row writes half a row at a row
    boundary and produces plausible garbage."""
    desc = strided_desc(0, 1, rows=4, run_bytes=SLOT + 1, spitch=SLOT * 2,
                        dpitch=SLOT * 2, src_ptr=dev_ptr(0, 0),
                        dst_ptr=dev_ptr(1, 0))
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        tp.batch_descs([desc], SLOT)
    assert "W68 Weg2XchgPlanDisagree" in str(excinfo.value)
    assert "run_bytes" in str(excinfo.value)


def test_an_unknown_copy_kind_is_refused_not_treated_as_flat():
    desc = flat_desc(0, 1, 64, src_ptr=dev_ptr(0, 0), dst_ptr=dev_ptr(1, 0))
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        tp.batch_descs([desc.replace(kind="RESHAPE2D")], SLOT)
    assert "RESHAPE2D" in str(excinfo.value)


def test_a_batch_never_exceeds_the_slot_it_will_be_written_into():
    """The one arithmetic invariant the whole handshake rests on.

    ``publish`` refuses a ``bytes_filled`` over ``SLOT_BYTES``, so a batcher
    that over-packed would turn every flip into a W68 -- but only after the
    producer had already issued the copies past the end of the slot, into the
    NEXT pair's slot.  The check belongs before the bytes move.
    """
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    descs = [flat_desc(0, 1, 3000, src_off=i * 3000, dst_off=i * 3000,
                       src_ptr=src, dst_ptr=dst, name=f"p{i}")
             for i in range(7)]
    descs.append(strided_desc(0, 1, rows=40, run_bytes=300, spitch=512,
                              dpitch=1024, src_off=30000, dst_off=30000,
                              src_ptr=src, dst_ptr=dst))
    for batch in tp.batch_descs(descs, SLOT):
        assert batch.total_bytes <= SLOT
        assert sum(p.nbytes for p in batch.pieces) == batch.total_bytes
        for piece in batch.pieces:
            assert piece.slot_off + piece.nbytes <= SLOT
    # And nothing was dropped on the way.
    moved = sum(b.total_bytes for b in tp.batch_descs(descs, SLOT))
    assert moved == sum(d.nbytes for d in descs)


# ===========================================================================
# THE ON-CARD LANE.
# ===========================================================================


def test_oncard_two_hops_land_the_bytes_and_the_handle_survives(region, tmp_path):
    """Source VRAM -> its own bounce -> the imported peer pointer -> destination.

    Both hops are asserted, and so is the property that made S0 worth running:
    the consumer's pointer is NOT the producer's, and the 64-byte handle
    survives the region round trip with its trailing byte intact.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    cons = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(21, 5000)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(prod, src, payload)
        poison(cons, dst, len(payload), seed=0xCC)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]

        bounce = tp.OnCardBounce(prod, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        tp.publish_ipc_handle(region, 0, bounce.handle)
        assert tp.read_ipc_handle(region, 0) == bounce.handle
        assert tp.read_ipc_handle(region, 0)[-1] == 0xA5, "c_char would truncate"
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                            slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_ARMED)

        peer = cons.ipc_open_handle(tp.read_ipc_handle(region, 0))
        assert peer != bounce.ptr, "an imported IPC pointer is a DIFFERENT address"

        pstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        cstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        errors: list = []

        def produce():
            try:
                tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                       bounce, row=0, peer_row=3, wave=WAVE,
                                       descs=descs, stats=pstats,
                                       budget_s=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        tp.run_oncard_consumer(region, cons, cons.create_stream(0), peer,
                               row=3, peer_row=0, wave=WAVE, descs=descs,
                               stats=cstats, slots=tp.ONCARD_SLOTS,
                               slot_bytes=SLOT, budget_s=10.0)
        thread.join(30)
        assert not errors, errors
        assert read(cons, dst, len(payload)) == payload
        assert pstats.batches == cstats.batches > 1, \
            "the double buffer really cycled"
        bounce.close()
    finally:
        prod.close()
        cons.close()


def test_oncard_consumer_refuses_a_short_hop(region, tmp_path):
    """W70 on the diagonal too -- one refusal, both lanes.

    The on-card lane has no semaphore (the diagonal is absent from
    ``CROSS_PAIRS``), so it would have been easy to give it its own weaker
    check.  It gets the same one, with ``lane=oncard`` so a log reader can tell
    which lane refused.
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(23, 3000)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(ops, src, payload)
        poison(ops, dst, len(payload), seed=0xDD)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst)]
        batch = tp.batch_descs(descs, SLOT)[0]
        bounce = tp.OnCardBounce(ops, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=batch.seq,
                            nbytes=batch.total_bytes - 8,
                            slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_READY)
        before = ops.issued
        with pytest.raises(tp.Weg2XchgShortPiece) as excinfo:
            tp.run_oncard_consumer(region, ops, ops.create_stream(0), bounce.ptr,
                                   row=3, peer_row=0, wave=WAVE, descs=descs,
                                   stats=tp.OnCardStats(0, "u0", "ipc"),
                                   slots=tp.ONCARD_SLOTS, slot_bytes=SLOT,
                                   budget_s=2.0)
        assert tp.SHORT_PIECE_MARKER in str(excinfo.value)
        assert "lane=oncard" in str(excinfo.value)
        assert ops.issued == before
        assert read(ops, dst, len(payload)) != payload
        bounce.close()
    finally:
        ops.close()


def test_a_previous_flips_oncard_row_is_not_this_flips_signal(region, boot):
    """The wave gate's own stop-loss, one lane over.

    A row left at its last sequence by the previous flip satisfies the first
    poll of the next flip with nobody having filled anything.  The epoch hash
    is what stops it, so the epoch hash is what this test moves.
    """
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=5, nbytes=99,
                        slot_bytes=SLOT, wave=WAVE,
                        state=tp.ONCARD_STATE_READY)
    assert tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 5, 1.0,
                            what="fill", row=3, wave=WAVE)["bytes"] == 99
    region.begin_flip(f"{boot}.2")
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 5, 0.2,
                         what="fill", row=3, wave=WAVE)
    assert "denominator" in str(excinfo.value)
    assert "epoch_hash" in str(excinfo.value)
    assert "alive_in_proc" in str(excinfo.value)


def test_a_second_batch_does_not_overwrite_the_first_batchs_row(region):
    """THE MEASURED DEFECT of the first green remote run, pinned.

    With ONE row per rank, the producer's batch 1 overwrote batch 0's ``bytes``
    before the consumer read it, and the consumer -- which waits for a
    sequence and then compares byte counts -- raised W70 against a producer
    that had done nothing wrong (``expected_bytes=4096 bytes_filled=1904``:
    batch 1's number answering batch 0's question).  Two halves, and the second
    is the one that makes the first safe:

    * a row is per rank PER SLOT, so batch 0's description survives while
      batch 1 is being filled into the other slot;
    * the FILL wait is EXACT, so a later batch's row can never answer for an
      earlier one even if the layout changed again.
    """
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, slot=0, seq=0,
                        nbytes=4096, slot_bytes=SLOT, wave=WAVE,
                        state=tp.ONCARD_STATE_READY)
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, slot=1, seq=1,
                        nbytes=1904, slot_bytes=SLOT, wave=WAVE,
                        state=tp.ONCARD_STATE_READY)
    assert tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, 0)["bytes"] == 4096
    assert tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, 1)["bytes"] == 1904

    got = tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 1.0,
                           what="fill", row=3, slot=0, wave=WAVE, exact=True)
    assert got["bytes"] == 4096, "batch 0's row must still describe batch 0"

    # And exactness: slot 1 carries batch 1, so asking it about batch 0 is a
    # timeout, never batch 1's byte count.
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 0.2,
                         what="fill", row=3, slot=1, wave=WAVE, exact=True)
    assert "exact=yes" in str(excinfo.value)
    assert "peer_seq=1" in str(excinfo.value)
    # The DRAIN direction is monotone and must stay >=.
    assert tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 1.0,
                            what="drain", row=3, slot=1, wave=WAVE)["seq"] == 1


def test_run_leg_applies_its_slot_size_to_both_lanes():
    """A knob that covers half of what its name says is worse than no knob.

    The first version threaded ``slot_bytes`` into the cross lane only and left
    the on-card bounce on the module default, which the six-process double
    caught the hard way: a 4 KiB leg allocated a 64 MiB bounce and ran the fake
    device out of memory.  ``oncard_slot_bytes`` defaults to ``slot_bytes``, and
    the double asserts the allocation size that follows from it.
    """
    sig = inspect.signature(tp.run_leg)
    assert sig.parameters["oncard_slot_bytes"].default is None
    source = inspect.getsource(tp.run_leg)
    assert "slot_bytes if oncard_slot_bytes is None else oncard_slot_bytes" \
        in source
    # EVERY diagonal site, found by an AST walk rather than counted by
    # hand: a literal count is a test that has to be edited whenever a
    # site is added, which is precisely when it should fail instead.
    tree = ast.parse(source.lstrip())
    sized = {"OnCardBounce", "HostBounce", "run_oncard_consumer"}
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in sized:
            continue
        seen += 1
        kw = {k.arg: k.value for k in node.keywords}
        assert "slot_bytes" in kw, f"{name} takes the module default"
        assert getattr(kw["slot_bytes"], "id", None) == "diag_bytes", name
    assert seen >= 3, f"only {seen} sized call sites found in run_leg"


def test_a_torn_oncard_row_is_not_a_signal(region):
    """Half a row must not read as a whole one -- the same seal law as S3's
    gate rows, in the area S4 owns."""
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 1, seq=2, nbytes=64,
                        slot_bytes=SLOT, wave=WAVE,
                        state=tp.ONCARD_STATE_READY)
    assert tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 1)["sealed"] == 1
    view = region.dir_view()
    off = tp._oncard_row_off(tp.DIR_ONCARD_PROD_OFF, 1)
    ctypes.memmove(ctypes.addressof(view) + off + 8, struct.pack("<Q", 999), 8)
    assert tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 1)["sealed"] == 0
    with pytest.raises(xr.Weg2XchgGateTimeout):
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 1, 2, 0.2,
                         what="fill", row=4, wave=WAVE)


def test_an_armed_row_is_the_handle_signal_and_not_a_batch(region):
    """``ARMED``/``seq=-1`` means "the bounce exists", never "batch 0 is ready".

    Folding the two would make batch 0's wait also the handle's wait, and the
    consumer would then open a buffer it had never been told was there --
    or, worse, treat the arming row as a filled batch of zero bytes and raise
    W70 against a producer that had done nothing wrong.
    """
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                        slot_bytes=SLOT, wave=WAVE,
                        state=tp.ONCARD_STATE_ARMED)
    tp._await_handle(region, 0, 1.0, row=3, wave=WAVE)   # returns at once
    with pytest.raises(xr.Weg2XchgGateTimeout):          # but is not batch 0
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 0.2,
                         what="fill", row=3, wave=WAVE)


def test_await_handle_names_a_source_that_never_armed(region):
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_handle(region, 2, 0.2, row=5, wave=WAVE)
    message = str(excinfo.value)
    assert "what=handle" in message
    assert "peer_row=2" in message
    assert "never published a bounce handle" in message


def test_the_oncard_producer_may_not_run_ahead_of_the_double_buffer(region, tmp_path):
    """The diagonal's ONLY back-pressure, proven by making it bite.

    MEASURED-BY-REVIEW GAP (S4 review, 2026-09-09, must_fix, a SURVIVING
    MUTANT): deleting the whole in-loop ``_await_oncard(..., what="drain")``
    call left 36/36 green, because no on-card payload in the suite exceeded two
    slots -- ``batch.seq - bounce.slots`` was negative on every path and the
    call returned at its base case without ever reading a row.  The lane is
    double-buffered precisely so the producer can be held; nothing proved it
    ever was.

    The kill is a COUNT, not a timeout: with a consumer that drains nothing,
    a producer that respects the double buffer fills exactly
    ``ONCARD_SLOTS`` batches and then blocks.  Without the wait it fills all
    four, overwriting slot 0's row while batch 0 is undrained -- the same
    overwrite class that was already fixed once, one wait over.
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(29, 4 * SLOT - 100)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(ops, src, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]
        assert len(tp.batch_descs(descs, SLOT)) == 4, "not deep enough to block"
        bounce = tp.OnCardBounce(ops, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            tp.run_oncard_producer(region, ops, ops.create_stream(0), bounce,
                                   row=0, peer_row=3, wave=WAVE, descs=descs,
                                   stats=stats, budget_s=0.3)
        assert "what=drain " in str(excinfo.value), str(excinfo.value)
        assert stats.batches == tp.ONCARD_SLOTS, (
            f"the producer filled {stats.batches} slots into a "
            f"{tp.ONCARD_SLOTS}-slot bounce with nobody draining -- batch "
            f"{tp.ONCARD_SLOTS}'s row overwrote batch 0's while batch 0 was "
            f"still undrained")
        bounce.close()
    finally:
        ops.close()


def test_the_oncard_producer_does_not_return_before_the_last_batch_is_drained(
        region, tmp_path):
    """The TERMINAL half of the lifetime handshake, on its own.

    The in-loop drain wait stops the producer refilling a slot; it says nothing
    about the LAST ``ONCARD_SLOTS`` batches, which no later batch ever waits
    for.  Those are exactly the ones still in flight when the caller frees the
    bounce, so ``run_oncard_producer`` may not return while any of them is
    undrained.  A consumer that stops one batch short must therefore hold the
    producer to its budget, not let it through.
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(53, 3 * SLOT - 30)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(ops, src, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]
        batches = tp.batch_descs(descs, SLOT)
        assert len(batches) == 3
        # The consumer drained everything EXCEPT the last batch.
        for batch in batches[:-1]:
            tp.write_oncard_row(region, tp.DIR_ONCARD_CONS_OFF, 3,
                                slot=batch.seq % tp.ONCARD_SLOTS, seq=batch.seq,
                                nbytes=batch.total_bytes, slot_bytes=SLOT,
                                wave=WAVE, state=tp.ONCARD_STATE_DONE)
        bounce = tp.OnCardBounce(ops, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            tp.run_oncard_producer(region, ops, ops.create_stream(0), bounce,
                                   row=0, peer_row=3, wave=WAVE, descs=descs,
                                   stats=stats, budget_s=0.3)
        assert "what=drain-final" in str(excinfo.value), str(excinfo.value)
        assert f"seq={batches[-1].seq}" in str(excinfo.value)
        assert stats.batches == len(batches), (
            "it must fail at the END, having filled every batch -- a failure "
            "earlier than that is the in-loop wait, not this one")
        bounce.close()
    finally:
        ops.close()


def test_the_exported_bounce_is_not_freed_until_the_peer_has_released_it(
        tmp_path, boot, sems):
    """A ``cudaFree`` under a peer's open IPC mapping is the silent-wrongness
    class at its worst, and the fake could not see it.

    MEASURED-BY-REVIEW DEFECT (S4 review + refuter, 2026-09-09, must_fix):
    ``run_oncard_producer`` returned as soon as its last row was written and
    ``run_leg``'s ``finally`` freed the bounce, while the co-located consumer
    could still be issuing D2D out of the imported pointer for up to
    ``ONCARD_SLOTS`` batches -- 64 MiB per card per wave at the shipping
    default, wrong bytes, no rc, nothing raised.  CUDA also forbids the
    EXPORTER freeing an allocation an importer still has mapped, which the
    terminal drain alone does not cover: "your last copy landed" and "your
    mapping is gone" are two different facts.

    Six green tests, thirty-six green assertions and six mutants missed it
    because the fake's ``raw_free`` was ``return None`` -- the double DID
    execute the racing free and could not perceive it.  ``raw_free`` poisons
    now, and this test pins the ORDER against a consumer that deliberately
    holds its mapping open after its last byte has landed.
    """
    events: list = []
    lock = threading.Lock()

    def note(what: str) -> None:
        with lock:
            events.append((what, time.perf_counter()))

    region = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0,
                         on_free=lambda _p, _n: note("free"))
    cons = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        region.begin_flip(f"{boot}.1")
        payload = pattern(37, 3 * SLOT - 50)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(prod, src, payload)
        poison(cons, dst, len(payload), seed=0x4D)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]
        errors: list = []

        def consume() -> None:
            try:
                tp._await_handle(region, 0, 20.0, row=3, wave=WAVE)
                peer = cons.ipc_open_handle(tp.read_ipc_handle(region, 0))
                tp.run_oncard_consumer(
                    region, cons, cons.create_stream(0), peer, row=3,
                    peer_row=0, wave=WAVE, descs=descs,
                    stats=tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC),
                    slots=tp.ONCARD_SLOTS, slot_bytes=SLOT, budget_s=20.0)
                # Every byte has landed and every CONS row is written, so the
                # producer's terminal DRAIN is satisfied here -- and the
                # mapping is still open.  This window is the defect.
                time.sleep(0.3)
                cons.ipc_close_handle(peer)
                note("release")
                tp.write_oncard_release(region, 3, wave=WAVE)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                tp.write_oncard_release(region, 3, wave=WAVE)

        thread = threading.Thread(target=consume)
        thread.start()
        try:
            tp.run_leg(region, sems, prod, row=0, rank=0, device=0,
                       card_uuid="GPU-0",
                       uuid_of_card=[f"GPU-{c}" for c in range(xr.N_CARDS)],
                       descs=descs, is_source=True,
                       oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
                       log=lambda _ln: None,
                       vote_failure=lambda exc: None,
                       budget_s=20.0, slot_bytes=SLOT)
        finally:
            thread.join(60)
        assert not errors, errors
        assert read(cons, dst, len(payload)) == payload

        order = [what for what, _t in sorted(events, key=lambda e: e[1])]
        assert order.count("free") == 1, (
            f"the bounce must be freed exactly once, got {events}")
        assert order == ["release", "free"], (
            f"the exporter freed its bounce while the importer still had it "
            f"mapped: {events}")
        assert prod.freed and prod.freed[0][1] == tp.ONCARD_SLOTS * SLOT
    finally:
        prod.close()
        cons.close()
        region.close()


def test_a_previous_waves_oncard_row_is_not_this_waves_signal(region):
    """The epoch hash is per FLIP; ``run_leg`` is per WAVE.

    MEASURED-BY-REVIEW DEFECT (S4 refuter, 2026-09-09, must_fix): every wave
    restarts ``seq`` at 0 with a new bounce and a new handle, and the row
    carried no wave stamp -- so at the SAME epoch hash, wave 1's leftover rows
    answered wave 2's questions.  All three waits were affected and they fail
    differently, so all three are asserted:

    * DRAIN, always reachable: wave 1's consumer row sits at a large sequence,
      so wave 2's ``>=`` returns instantly and the producer refills a slot
      nobody has drained -- a silent overwrite on every multi-wave flip.
    * FILL, reachable whenever a wave's on-card share is two batches or fewer:
      the stale row matches EXACTLY and either raises W70 against a healthy
      producer or copies out of an unfilled bounce.
    * HANDLE: waves 2 and 3 never wait for the new handle at all and read
      wave 1's 64 bytes -- of a buffer the previous wave has already freed.

    The only staleness test this file had moved ``begin_flip``, which is the
    coarser stamp; nothing here ever called ``run_leg`` twice.
    """
    # -- DRAIN ------------------------------------------------------------
    tp.write_oncard_row(region, tp.DIR_ONCARD_CONS_OFF, 3, slot=0, seq=110,
                        nbytes=SLOT, slot_bytes=SLOT, wave=0,
                        state=tp.ONCARD_STATE_DONE)
    assert tp._await_oncard(region, tp.DIR_ONCARD_CONS_OFF, 3, 0, 1.0,
                            what="drain", row=0, slot=0, wave=0)["seq"] == 110
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_oncard(region, tp.DIR_ONCARD_CONS_OFF, 3, 0, 0.2,
                         what="drain", row=0, slot=0, wave=1)
    assert "peer_wave=0" in str(excinfo.value)
    assert "wave=1" in str(excinfo.value)
    assert "and wave=1)" in str(excinfo.value), "the denominator names the wave"

    # -- FILL -------------------------------------------------------------
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, slot=0, seq=0,
                        nbytes=SLOT, slot_bytes=SLOT, wave=0,
                        state=tp.ONCARD_STATE_READY)
    assert tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 1.0,
                            what="fill", row=3, slot=0, wave=0,
                            exact=True)["bytes"] == SLOT
    with pytest.raises(xr.Weg2XchgGateTimeout):
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 0.2,
                         what="fill", row=3, slot=0, wave=1, exact=True)

    # -- HANDLE -----------------------------------------------------------
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, slot=0, seq=-1,
                        nbytes=0, slot_bytes=SLOT, wave=0,
                        state=tp.ONCARD_STATE_ARMED)
    tp._await_handle(region, 0, 1.0, row=3, wave=0)
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_handle(region, 0, 0.2, row=3, wave=1)
    assert "peer_wave=0" in str(excinfo.value)
    assert "what=handle" in str(excinfo.value)


def test_the_two_sides_must_agree_on_the_oncard_slot_size(region, tmp_path):
    """``slot_bytes`` decides byte placement and was agreed by nothing.

    Two co-located ranks entering with different batcher slot sizes produce
    identical batches for every payload below the smaller of the two and
    diverge silently above it -- and on this lane there is no ``bytes_filled``
    record to catch it, only the row.  So the row carries the producer's
    geometry and the consumer refuses the disagreement by name, before the
    first copy (#802 rule 4: a number two ranks derive independently must be
    handshaken, not assumed).
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(43, 3000)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(ops, src, payload)
        poison(ops, dst, len(payload), seed=0x5E)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst)]
        batch = tp.batch_descs(descs, SLOT)[0]
        # A payload this size batches identically at either slot size, so the
        # byte count agrees and ONLY the geometry stamp can refuse it.
        assert tp.batch_descs(descs, 2 * SLOT)[0].total_bytes == batch.total_bytes
        bounce = tp.OnCardBounce(ops, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, slot=0,
                            seq=batch.seq, nbytes=batch.total_bytes,
                            slot_bytes=2 * SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_READY)
        before = ops.issued
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            tp.run_oncard_consumer(region, ops, ops.create_stream(0), bounce.ptr,
                                   row=3, peer_row=0, wave=WAVE, descs=descs,
                                   stats=tp.OnCardStats(0, "u0", "ipc"),
                                   slots=tp.ONCARD_SLOTS, slot_bytes=SLOT,
                                   budget_s=2.0)
        assert f"slot_bytes={2 * SLOT}" in str(excinfo.value)
        assert ops.issued == before, "the refusal must precede the first copy"
        assert read(ops, dst, len(payload)) != payload
        bounce.close()
    finally:
        ops.close()


def test_a_slot_size_the_region_cannot_hold_is_refused_before_any_copy(
        region, sems, ops):
    """The one arithmetic bound that was stated and never enforced.

    MEASURED-BY-REVIEW DEFECT (S4 review + refuter, 2026-09-09, must_fix):
    ``batch_descs`` refused only ``slot_bytes <= 0`` and both pair runners
    accepted whatever they were handed, while the staging slots are exactly
    ``xr.SLOT_BYTES`` apart.  A larger value issues copies past the end of the
    slot into the NEXT directed pair's live payload; ``publish``'s
    ``bytes_filled > SLOT_BYTES`` refusal is the only guard and it fires after
    the bytes have landed.  ``test_a_batch_never_exceeds_the_slot_it_will_be_
    written_into`` names this exact hazard in its docstring and proves only
    that the batcher respects the number it was given -- the half that was
    never in doubt.
    """
    over = xr.SLOT_BYTES + 1
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    descs = [flat_desc(0, 1, 512, src_ptr=src, dst_ptr=dst)]
    before = ops.issued
    for runner in (tp.run_producer_pair, tp.run_consumer_pair):
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            runner(region, sems, ops, ops.create_stream(0),
                   pair=xr.pair_id(0, 1), descs=descs,
                   stats=tp.PairStats(0, 1, "a", "b"), budget_s=1.0,
                   slot_bytes=over)
        assert "W68 Weg2XchgPlanDisagree" in str(excinfo.value)
        assert f"slot_bytes={over}" in str(excinfo.value), runner.__name__
        assert "next pair's live payload" in str(excinfo.value)
    assert ops.issued == before, "the refusal must precede the first copy"

    # Both of run_leg's knobs, at the door, before a thread exists.
    votes: list = []

    def leg(**over_kw):
        return tp.run_leg(region, sems, ops, row=0, rank=0, device=0,
                          card_uuid="GPU-0",
                          uuid_of_card=[f"GPU-{c}" for c in range(xr.N_CARDS)],
                          descs=descs, is_source=True,
                          oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
                          log=lambda _ln: None, vote_failure=votes.append,
                          budget_s=1.0, **over_kw)

    with pytest.raises(xr.Weg2XchgPlanDisagree):
        leg(slot_bytes=over)
    # S5 MOVED THIS CEILING, and the property is unchanged: each knob is
    # bounded by the storage it writes into.  The cross knob writes into the
    # region's slots (xr.SLOT_BYTES); the diagonal writes into a bounce this
    # module allocates itself, whose ceiling became ONCARD_SLOT_BYTES_MAX when
    # the slot size started being PRICED from the plan's batch count (S5-pre
    # measured the lane per-batch-bound, so the slot is a decision, not a
    # constant).
    with pytest.raises(xr.Weg2XchgPlanDisagree):
        leg(slot_bytes=SLOT, oncard_slot_bytes=tp.ONCARD_SLOT_BYTES_MAX + 1)
    assert votes == [], (
        "a knob refused at the door is not a leg that failed mid-flight; "
        "voting ok=False here would take the whole group down for a caller's "
        "argument error")
    assert ops.issued == before

    # The ceiling is the region's OWN header value, not a module constant this
    # function happens to agree with.
    assert region.header()["slot_bytes"] == xr.SLOT_BYTES
    assert tp.require_slot_bytes(xr.SLOT_BYTES, xr.SLOT_BYTES,
                                 what="t") == xr.SLOT_BYTES
    for bad in (0, -1):
        with pytest.raises(xr.Weg2XchgPlanDisagree):
            tp.require_slot_bytes(xr.SLOT_BYTES, bad, what="t")


def test_the_sequence_half_of_the_short_piece_check_fires_on_its_own(
        region, sems, ops):
    """W70's ``rec.seq != batch.seq`` half, alone.

    MEASURED-BY-REVIEW GAP (S4 review, 2026-09-09, a SURVIVING MUTANT):
    reducing the check to its byte-count half left 36/36 green.  On the real
    flip that is the half that never fires -- nearly every batch is a full
    32 MiB slot, so ``bytes_filled == total_bytes`` by construction -- and
    ``claim_produced`` checks the epoch and never the sequence, which makes
    this the ONLY sequence discriminator in the design.  A slot carrying a
    different batch of the same size would be copied without a word.
    """
    pair = xr.pair_id(0, 1)
    payload = pattern(47, 700)
    src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
    write(ops, src, payload)
    poison(ops, dst, len(payload), seed=0x6F)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst)]
    batch = tp.batch_descs(descs, SLOT)[0]

    # The producer publishes the RIGHT byte count under the WRONG sequence.
    assert sems.trywait(pair, batch.slot, "empty")
    region.begin_fill(pair, batch.slot, batch.seq + 4)
    stream = ops.create_stream(0)
    ops.memcpy_async(region.slot_address(pair, batch.slot), src, len(payload),
                     stream)
    ops.synchronize(stream)
    region.publish(pair, batch.slot, batch.total_bytes)
    sems.post(pair, batch.slot, "full")

    before = ops.issued
    with pytest.raises(tp.Weg2XchgShortPiece) as excinfo:
        tp.run_consumer_pair(region, sems, ops, ops.create_stream(0),
                             pair=pair, descs=descs,
                             stats=tp.PairStats(0, 1, "u0", "u1"),
                             budget_s=2.0, slot_bytes=SLOT)
    message = str(excinfo.value)
    assert tp.SHORT_PIECE_MARKER in message
    assert "short_by=0" in message, "the byte counts AGREE -- that is the point"
    assert f"seq_filled={batch.seq + 4}" in message
    assert f"seq={batch.seq} " in message
    assert "seq_ok=no" in message, (
        "a W70 that printed only byte counts would read as a contradiction of "
        "itself here")
    assert ops.issued == before
    assert read(ops, dst, len(payload)) != payload


class _ScriptedSems(tp.SemSet):
    """A :class:`tp.SemSet` whose two RAW syscalls are scripted.

    Nothing else is replaced: the retry logic, the deadline arithmetic and the
    errno classification under test are the product's own.  ``handle`` is
    overridden so no named semaphore is opened -- the scripted calls never
    dereference it.  The script is POPPED, so a runaway retry loop raises
    ``IndexError`` instead of hanging the suite.
    """

    def __init__(self, boot_nonce: str, script):
        super().__init__(boot_nonce)
        self.script = list(script)
        self.calls = 0
        self.deadlines: list = []

    def handle(self, pair, slot, kind):
        return 0xDEAD

    def _timedwait_once(self, handle, ts):
        self.calls += 1
        self.deadlines.append((ts.tv_sec, ts.tv_nsec))
        return self.script.pop(0)

    def _trywait_once(self, handle):
        self.calls += 1
        return self.script.pop(0)


def test_a_signal_during_a_slot_wait_is_not_a_timeout(boot):
    """EINTR read as a timeout is a group-fatal refusal on a healthy peer.

    MEASURED-BY-REVIEW DEFECT (S4 refuter, 2026-09-09, must_fix):
    ``sem_timedwait``'s return code was compared against 0 and errno was never
    read, although the CDLL was opened with ``use_errno=True``.  POSIX permits
    ``EINTR``, and these waits run inside a live ``launch_server`` beside
    torch's watchdogs and child reaping -- so a signal produced
    ``W69 ... no producer posted this slot within the fence budget`` naming a
    rank that was perfectly healthy, and ``run_leg`` then voted the group down
    (W29 -> front W4 -> do_stop).  A guard that cries wolf on the healthy path
    is the failure mode this file names in
    ``test_batches_are_the_same_whether_the_side_coalesced_or_not``.

    Three classifications and one arithmetic property: EINTR retries,
    ETIMEDOUT is the only false, any other errno raises rather than being
    silently swallowed -- and the retry uses the SAME ABSOLUTE DEADLINE, so a
    storm of signals cannot extend the fence budget.
    """
    interrupted = _ScriptedSems(boot, [errno.EINTR, errno.EINTR, errno.EINTR, 0])
    assert interrupted.timedwait(0, 0, "full", 5.0) is True
    assert interrupted.calls == 4
    assert len(set(interrupted.deadlines)) == 1, (
        f"the deadline was rebuilt inside the retry loop, so every signal "
        f"extends the fence budget: {interrupted.deadlines}")

    timed_out = _ScriptedSems(boot, [errno.ETIMEDOUT])
    assert timed_out.timedwait(0, 0, "full", 0.01) is False

    broken = _ScriptedSems(boot, [errno.EINVAL])
    with pytest.raises(OSError) as excinfo:
        broken.timedwait(0, 0, "full", 0.01)
    assert excinfo.value.errno == errno.EINVAL

    # The non-blocking probe classifies the same way: EAGAIN is "not available"
    # (which is what makes slot_waits exact), EINTR is not.
    poll = _ScriptedSems(boot, [errno.EINTR, errno.EAGAIN])
    assert poll.trywait(0, 0, "empty") is False
    assert poll.calls == 2
    ready = _ScriptedSems(boot, [0])
    assert ready.trywait(0, 0, "empty") is True


def test_every_thread_that_failed_is_reported_not_only_the_first(
        region, sems, ops):
    """Two threads die; the one that is raised must not erase the other.

    ``run_leg`` used to call ``vote_failure(errors[0])`` and raise it, dropping
    the rest.  When a pair thread and the diagonal fail together the survivor
    may name the consequence rather than the cause, and the other error would
    then exist nowhere -- not in the log, not on the exception.
    """
    src = dev_ptr(0, 0)
    descs = []
    for dst_card in (1, 2):
        descs.append(flat_desc(0, dst_card, 3 * SLOT - 40,
                               src_off=dst_card * 0x40000,
                               dst_off=dst_card * 0x40000,
                               src_ptr=src, dst_ptr=dev_ptr(dst_card, 0),
                               name=f"p{dst_card}"))
    lines: list = []
    votes: list = []
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp.run_leg(region, sems, ops, row=0, rank=0, device=0,
                   card_uuid="GPU-0",
                   uuid_of_card=[f"GPU-{c}" for c in range(xr.N_CARDS)],
                   descs=descs, is_source=True,
                   oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
                   log=lines.append, vote_failure=votes.append,
                   budget_s=0.3, slot_bytes=SLOT)
    assert len(votes) == 1, "exactly one vote, whatever the thread count"
    assert votes[0] is excinfo.value
    extra = [ln for ln in lines if "also failed" in ln]
    assert len(extra) == 1, lines
    assert "thread 2 of 2" in extra[0], extra[0]
    assert "W69 Weg2XchgGateTimeout" in extra[0]
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("also failed" in n for n in notes), notes


# ===========================================================================
# REGISTRATION, LAYOUT AND THE INTERFACES S5/S6 MUST NOT DRIFT FROM.
# ===========================================================================


def test_a_failed_host_register_is_not_counted_as_registered(region, tmp_path):
    """The denominator is the instrument, so it must not lie.

    A rank whose ``cudaHostRegister`` failed keeps the region usable at a
    measured 7.5 % / 3.3 % cost -- so this is not a refusal -- but it must not
    move ``registered=<n>/6``, or an armed region reads as pinned on six ranks
    while it is pinned on five.
    """
    bad = FakeDeviceOps(str(tmp_path / "d"), rank=0, fail_host_register=True)
    good = FakeDeviceOps(str(tmp_path / "d"), rank=1)
    try:
        lines: list = []
        result = tp.register_region(region, bad, 0, log=lines.append)
        assert result.registered is False
        assert region.registered_count() == 0
        assert tp.HOSTREG_LINE_PREFIX in lines[0] and "registered=no" in lines[0]
        assert "cudaHostRegister" in lines[0]
        assert f"count=0/{xr.N_RANKS}" in lines[0]

        lines.clear()
        result = tp.register_region(region, good, 1, log=lines.append)
        assert result.registered is True
        assert region.registered_count() == 1
        assert good.registered == [(region.base_address(), xr.REGION_BYTES,
                                    tp.CUDA_HOST_REGISTER_PORTABLE)]
        assert any("registered=yes" in ln for ln in lines)
        tp.unregister_region(good, result)
        assert good.registered == []
    finally:
        bad.close()
        good.close()


def test_the_dir_sub_layout_is_disjoint_and_inside_the_region():
    """S3 owns the region; S4 owns these bytes.  Both halves are checkable.

    The pointer table's 128 KiB is RESERVED, not used: S6 lands in a hole that
    was accounted for rather than one it has to be told about.
    """
    areas = [
        ("ptrtable", tp.DIR_PTRTABLE_OFF, tp.DIR_PTRTABLE_BYTES),
        ("handles", tp.DIR_HANDLE_OFF, tp.DIR_HANDLE_BYTES),
        ("oncard_prod", tp.DIR_ONCARD_PROD_OFF, tp.DIR_ONCARD_PROD_BYTES),
        ("oncard_cons", tp.DIR_ONCARD_CONS_OFF, tp.DIR_ONCARD_CONS_BYTES),
        ("oncard_rel", tp.DIR_ONCARD_REL_OFF, tp.DIR_ONCARD_REL_BYTES),
    ]
    cursor = 0
    for name, off, size in areas:
        assert off == cursor, f"{name} does not abut its predecessor"
        cursor = off + size
    assert cursor == tp.DIR_USED_BYTES <= tp.DIR_CAPACITY
    assert tp.DIR_CAPACITY == xr.DATA_OFF - xr.DIR_OFF
    # S5 appended `checksum` to the row: the diagonal has no slot RECORD to
    # carry it (xr.CROSS_PAIRS has no diagonal), so the row is the only channel
    # a co-located producer has.  The row still fits its 128-byte cell with the
    # seal.
    assert tp.ONCARD_ROW_STRUCT.size + 8 <= tp.DIR_ONCARD_ROW_BYTES
    # ONE ROW PER RANK PER SLOT -- see the measured overwrite defect.
    # S5: sized from ONCARD_SLOTS_MAX, not from the shipping default, because
    # the pipeline depth became a per-plan knob and two processes compute these
    # offsets independently -- a row area that followed an argument would put
    # rank r's slot k on rank r+1's slot 0, identically on both sides.
    assert tp.DIR_ONCARD_ROWS == xr.N_RANKS * tp.ONCARD_SLOTS_MAX
    assert tp.ONCARD_SLOTS <= tp.ONCARD_SLOTS_MAX
    assert tp.DIR_ONCARD_PROD_BYTES == tp.DIR_ONCARD_ROWS * tp.DIR_ONCARD_ROW_BYTES
    # The row carries seq, bytes, slot_bytes, wave, epoch_hash, pid, state and
    # (S5) checksum -- and its seal still fits inside the row with room left
    # over, so the next field to be added does not silently overwrite the seal.
    # That headroom is what let S5 add one without moving the row size, which
    # is the property this line grades rather than the count itself.
    assert tp.ONCARD_ROW_STRUCT.size == 8 + 7 * 8
    assert tp.ONCARD_SEAL_OFF + 8 < tp.DIR_ONCARD_ROW_BYTES
    with pytest.raises(ValueError):
        tp._oncard_row_off(tp.DIR_ONCARD_PROD_OFF, 0, tp.ONCARD_SLOTS_MAX)
    # ~1608 x 48 B is what spec 2.2 says the S6 pointer table needs.
    assert tp.DIR_PTRTABLE_BYTES >= 1608 * 48


def test_the_dir_view_cannot_reach_the_gate_rows_or_a_slot(region):
    """The bound is the reason S3 hands back this array rather than ``_mm``."""
    view = region.dir_view()
    assert len(view) == tp.DIR_CAPACITY
    with pytest.raises(IndexError):
        view[len(view)] = 1
    header_before = region.header()
    gate_before = region.read_gate_row(0)
    matrix_before = region.read_matrix_row(0)
    ctypes.memset(ctypes.addressof(view), 0x5A, len(view))
    assert region.header() == header_before
    for attr in ("gate_seq", "ok", "pid", "state", "sealed"):
        assert getattr(region.read_gate_row(0), attr) == getattr(gate_before, attr)
    assert region.read_matrix_row(0).sealed == matrix_before.sealed
    assert region.read_slot(0, 0).state == xr.SLOT_FREE


def test_the_fake_never_mistakes_a_host_address_for_a_device_one(region, ops):
    """A HARNESS test, kept because the harness got this wrong on the metal.

    First remote run of this file: the split was ``ptr < FAKE_DEV_BASE`` with
    the base at ``1 << 46``, and every real host address -- the mapped
    region's ``0x7f...`` -- sorted as a DEVICE pointer, was decoded into a rank
    whose file had never been sized for that offset, and the first staging copy
    segfaulted.  A harness that mis-sorts pointers cannot prove anything about
    a transport whose whole job is to move them, so the split is pinned here.

    The tag sits above the 47-bit x86-64 user range, which is what makes the
    real addresses below it and the assertion meaningful rather than lucky.
    """
    for host in (region.base_address(), region.slot_address(0, 0),
                 region.dir_address(), id(ops)):
        assert host < (1 << 48), f"{host:#x} is outside the user range"
        assert FakeDeviceOps.decode(host) is None, f"{host:#x} read as device"
        assert ops.real(host) == host

    for rank in range(xr.N_RANKS):
        for off in (0, 1, FAKE_DEV_BYTES - 1):
            ptr = dev_ptr(rank, off)
            assert FakeDeviceOps.decode(ptr) == (rank, off)
            assert FakeDeviceOps.decode(ptr | FAKE_IMPORT_BIAS) == (rank, off)
            assert ptr | FAKE_IMPORT_BIAS != ptr


def test_no_byte_moves_before_the_flip_is_bound(tmp_path, boot, sems):
    """THE GATE COMES BEFORE THE FIRST BYTE, on both lanes.

    Gate 0 runs in the RPC preamble, before any ``resume`` and before any
    ``pause``, and ``begin_flip`` is what stamps this flip's identity onto the
    slots and the rows.  A transport that ran without it would fill slots
    carrying the PREVIOUS flip's epoch hash -- which
    ``XchgRegion.claim_produced`` then zeroes as foreign, so the bytes would be
    silently dropped rather than loudly refused.

    Both lanes are asserted, because the diagonal has no semaphore and could
    easily have been given a weaker precondition.
    """
    unbound = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    ops_ = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        assert unbound.epoch == "" and unbound.epoch_hash == 0
        src, dst = dev_ptr(0, 0), dev_ptr(1, 0)
        descs = [flat_desc(0, 1, 512, src_ptr=src, dst_ptr=dst)]
        before = ops_.issued
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            tp.run_producer_pair(unbound, sems, ops_, ops_.create_stream(0),
                                 pair=xr.pair_id(0, 1), descs=descs,
                                 stats=tp.PairStats(0, 1, "a", "b"),
                                 budget_s=2.0, slot_bytes=SLOT)
        assert "W68 Weg2XchgPlanDisagree" in str(excinfo.value)
        assert "no flip is bound" in str(excinfo.value)
        assert ops_.issued == before, "the refusal must precede the first copy"

        with pytest.raises(xr.Weg2XchgPlanDisagree):
            tp.write_oncard_row(unbound, tp.DIR_ONCARD_PROD_OFF, 0, slot=0,
                                seq=0, nbytes=1, slot_bytes=SLOT, wave=WAVE,
                                state=tp.ONCARD_STATE_READY)
    finally:
        ops_.close()
        unbound.close()


def test_kind_constants_match_the_plan_builder():
    """The transport declares its own kind strings so it stays importable
    without torch; this is what keeps the two sets from drifting into a
    STRIDED2D descriptor silently taking the FLAT branch."""
    assert (tp.FLAT, tp.STRIDED2D, tp.ZEROFILL) == (wx.FLAT, wx.STRIDED2D, wx.ZEROFILL)


def test_slot_geometry_is_s3s_and_not_a_second_copy():
    assert tp.ONCARD_SLOT_BYTES == xr.SLOT_BYTES
    assert xr.SLOTS_PER_PAIR == 2
    assert [tp.SlotBatch(i, (), 0).slot for i in range(5)] == [0, 1, 0, 1, 0]


def test_the_transport_reads_the_fence_budget_and_never_a_literal():
    """No second timeout constant (spec 3.2).  Every wait here defaults to
    ``WEG2_GROUP_FENCE_BUDGET_S``; a numeric default would be a second budget
    that nobody would find when the first one was tuned."""
    for name in ("run_producer_pair", "run_consumer_pair", "run_oncard_producer",
                 "run_oncard_consumer", "run_leg"):
        sig = inspect.signature(getattr(tp, name))
        assert sig.parameters["budget_s"].default is None, name
    assert "xr.fence_budget_s() if budget_s is None" in inspect.getsource(tp)


def test_run_leg_interface_is_what_s6_must_call():
    """TODO(S6) is unwired on purpose; the SEAM is pinned anyway.

    S6 owns the RPC handler.  Until it lands nothing calls :func:`tp.run_leg`,
    and an unwired function is exactly the kind that drifts.  Every argument
    here is one S6 must supply, and ``vote_failure`` is REQUIRED rather than
    defaulted because an opt-in unwind is the weg2rg2 hold-and-wait.

    ``wave`` is REQUIRED for the same class of reason: this function is called
    once per wave, every on-card handshake row carries the wave, and a default
    would let wave 2 silently inherit wave 1's rows and wave 1's IPC handle at
    the same epoch hash.
    """
    sig = inspect.signature(tp.run_leg)
    required = {n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty}
    assert required == {
        "region", "sems", "ops", "row", "rank", "device", "card_uuid",
        "uuid_of_card", "descs", "is_source", "oncard_mode", "peer_row",
        "wave", "log", "vote_failure",
    }
    assert "TODO(S6)" in inspect.getdoc(tp.run_leg)
    assert "TODO(S5)" in inspect.getdoc(tp.SlotBatch.checksum)
    assert "TODO(S6)" in inspect.getdoc(tp.arm_oncard_lane)


def test_the_semaphores_are_opened_without_o_creat(boot):
    """A creating ``sem_open`` in a rank adopts a name with an unknown count,
    invisibly -- POSIX ignores the initial value for an existing name.  The
    launcher creates all 24 with ``O_EXCL``; this side must fail loudly when
    they are absent instead of manufacturing them."""
    source = inspect.getsource(tp.SemSet)
    assert "O_CREAT" in source
    assert 'sem_open(name.encode("ascii"), 0)' in source
    absent = tp.SemSet(f"{boot}-absent")
    with pytest.raises(OSError) as excinfo:
        absent.handle(0, 0, "empty")
    assert "O_CREAT" in str(excinfo.value)


def test_the_acceptance_lines_carry_every_token_the_spec_names():
    """Spec section 6 / S4's two acceptance lines, token by token.

    A line that has quietly lost a field is an acceptance that cannot be
    checked by the grep the record will actually run.
    """
    stats = tp.PairStats(0, 1, "GPU-src", "GPU-dst", bytes_moved=3 * xr.MIB,
                         strided_bytes=xr.MIB, pieces=17, slot_waits=2,
                         slot_wait_s=0.004, elapsed_s=0.5)
    line = stats.line()
    for token in ("WEG2-XCHG-PAIR", "src=GPU-src", "dst=GPU-dst",
                  "bytes_mib=3.00", "pieces=17", "strided_mib=1.00", "ms=500.000",
                  "gbs=0.006", "slot_waits=2", "slot_wait_ms=4.000"):
        assert token in line, token

    zero = tp.PairStats(0, 1, "a", "b")
    assert "gbs=0.000" in zero.line() and "bytes_mib=0.00" in zero.line()

    oncard = tp.OnCardStats(1, "GPU-card", tp.ONCARD_MODE_IPC,
                            bytes_moved=2 * xr.MIB, batches=4, elapsed_s=0.013)
    for token in ("WEG2-XCHG-ONCARD", "card=GPU-card", "mode=ipc",
                  "bytes_mib=2.00", "batches=4", "hop_ms=13.000"):
        assert token in oncard.line(), token
    # The field counts BATCHES and says so.  It was added as `hops`, which the
    # lane has exactly two of by construction, so on the real flip it would
    # have printed ~329 for a two-hop lane -- a name stating something the
    # number is not, in the very field added to fix a blindness.
    assert "hops=" not in oncard.line(), oncard.line()

    # A kilobyte lane prints bytes_mib=0.00 and MUST still be distinguishable
    # from one that moved nothing -- the measured reason the count is on the
    # line at all.
    small = tp.OnCardStats(1, "GPU-card", tp.ONCARD_MODE_IPC,
                           bytes_moved=5000, batches=2, elapsed_s=0.001)
    assert "bytes_mib=0.00" in small.line() and "batches=2" in small.line()
    nothing = tp.OnCardStats(1, "GPU-card", tp.ONCARD_MODE_IPC)
    assert "bytes_mib=0.00" in nothing.line() and "batches=0" in nothing.line()


def test_every_w_code_this_slice_raises_is_free_and_named_once():
    """W70 and W72 are what spec section 7 assigns to S4, and the branch census
    at 2ee844f7b8 leaves exactly those two free below W76 (W67/52/53/55/58/60
    are held by S1/S2/S3/S7; W73 is S6's and W75 is S5's).

    ``test_weg2_wcode_uniqueness_1263`` is the authority and walks the whole
    tree; this asserts the local half, so a collision introduced here is named
    here rather than three files away.
    """
    assert tp.SHORT_PIECE_MARKER == "W70 Weg2XchgShortPiece"
    assert tp.ONCARD_UNAVAILABLE_MARKER == "W72 Weg2XchgOnCardUnavailable"
    found = set(re.findall(r"\b(W\d{1,2}[a-z]?)\s+(Weg2[A-Za-z0-9_]+)",
                           inspect.getsource(tp)))
    assert found == {
        ("W68", "Weg2XchgPlanDisagree"),
        ("W69", "Weg2XchgGateTimeout"),
        ("W70", "Weg2XchgShortPiece"),
        ("W72", "Weg2XchgOnCardUnavailable"),
        # S5 (S4-fix refusal C, an S6 must_fix carried forward): the per-leg
        # semaphore re-arm check lives in this module because SemSet does.
        # W78 is free in the branch census -- W76 was the highest assigned and
        # W77 is the shadow's.
        ("W78", "Weg2XchgSemaphoreNotRearmed"),
        # S6 (#1273): the store-and-forward deposit's refusal.  ENUMERATED,
        # not picked -- the census over the four roots of
        # ``test_weg2_wcode_uniqueness_1263`` at ``edbf7007c8`` returns 54
        # assigned codes with a maximum of W80, so the next free code is W81.
        # The gaps below that maximum (W5, W6, W13-15, W18, W19, W23, W24,
        # W27, W39, W73) are deliberately NOT reused: a retired number still
        # matches every grep of every old boot log, and this file's own
        # history is two collisions bought by picking a number.
        ("W81", "Weg2XchgDepositUnfundable"),
        # S6 fix E (#1273): the on-card slot CEILING flag's refusal.  W81 was
        # the census maximum when the comment above was written, so W82 is the
        # next free code -- enumerated, not picked, and no retired gap reused.
        ("W82", "Weg2XchgOncardSlotRefused"),
    }, found


def test_find_libcudart_prefers_cu13_because_s0_measured_it():
    """The spec says ``libcudart.so.12`` throughout; S0 measured cu13 on the
    metal.  The search ORDER is the correction, and it is load-bearing: the
    preload hook links ``libcudart.so.13`` and a rank that cannot find it dies
    exit 127 before ``main()`` (launcher.py:2906-2908).
    """
    source = inspect.getsource(tp.find_libcudart)
    assert source.index("cu13/lib/libcudart.so.13") < \
        source.index("cuda_runtime/lib/libcudart.so.12")
    # It either resolves something or says what it tried -- never a bare
    # "no CUDA", which reads as a hardware verdict for a path problem.
    try:
        got = tp.find_libcudart(venv="/nonexistent-venv-for-this-test")
        assert "libcudart" in got
    except RuntimeError as exc:
        assert "LD_PRELOAD" in str(exc) and "exit 127" in str(exc)


# ===========================================================================
# THE HERMETIC DOUBLE: six processes, no CUDA, byte identity end to end.
# ===========================================================================

#: Device-file offsets, chosen so no two ranges of the double overlap.  Both
#: ranks of a card share one fake device file (they share a card), so a
#: collision here would make a passing assertion mean nothing.
ONCARD_SRC = 0x10000
ONCARD_DST = 0x30000
CROSS_BASE = 0x40000
CROSS_STRIDE = 0x8000
ZEROFILL_AT = 0x100000

#: THE DOUBLE RUNS ``run_leg`` TWICE, and that is a defect-driven number.
#: ``run_leg``'s contract is per WAVE while the epoch hash is per FLIP, so
#: every on-card handshake row of wave 2 sits beside a leftover row of wave 1
#: at the same epoch -- and nothing in this file called ``run_leg`` twice, so
#: the whole class was invisible (S4 refuter, must_fix).  Two waves with the
#: destination re-poisoned in between is the end-to-end half of that fix; the
#: row-level half is
#: ``test_a_previous_waves_oncard_row_is_not_this_waves_signal``.
DOUBLE_WAVES = 2


def _cross_off(src: int, dst: int) -> int:
    return CROSS_BASE + (src * xr.N_CARDS + dst) * CROSS_STRIDE


def _plan_for_double():
    """Three cards, two cross classes per directed pair, plus the diagonal.

    Deliberately mixed -- FLAT and STRIDED2D on every cross pair, an on-card
    share on every card, one ZEROFILL -- because the failure this double exists
    to catch is a lane that works alone and deadlocks beside the others.

    THE ON-CARD SHARE IS DELIBERATELY DEEPER THAN THE DOUBLE BUFFER.  It was
    5000 B against a 4096 B slot -- two batches -- so ``batch.seq -
    bounce.slots`` was negative on every path of every run and the producer's
    drain wait, the diagonal's only back-pressure, returned at its base case
    without ever reading a row.  Deleting that wait outright left the whole
    suite green (S4 review, a surviving mutant).  Four batches make the double
    buffer actually cycle and actually block.
    """
    descs = []
    for src in range(xr.N_CARDS):
        for dst in range(xr.N_CARDS):
            if src == dst:
                descs.append(flat_desc(
                    src, dst, 4 * SLOT - 100,
                    src_off=ONCARD_SRC, dst_off=ONCARD_DST,
                    src_ptr=dev_ptr(src, 0), dst_ptr=dev_ptr(dst, 0),
                    name=f"oncard.{src}"))
                continue
            off = _cross_off(src, dst)
            descs.append(flat_desc(
                src, dst, 6000, src_off=off, dst_off=off,
                src_ptr=dev_ptr(src, 0), dst_ptr=dev_ptr(dst, 0),
                name=f"flat.{src}{dst}"))
            descs.append(strided_desc(
                src, dst, rows=10, run_bytes=96, spitch=128, dpitch=320,
                src_off=off + 0x4000, dst_off=off + 0x4000 + 64,
                src_ptr=dev_ptr(src, 0), dst_ptr=dev_ptr(dst, 0),
                name=f"strided.{src}{dst}"))
    descs.append(wx.XchgDesc(
        tag="weights", src_rank=-1, dst_rank=2, param_name="vocab_pad",
        kind=tp.ZEROFILL, nbytes=256, rows=1, run_bytes=256, spitch=0, dpitch=0,
        dst_off=ZEROFILL_AT, dst_ptr=dev_ptr(2, 0)))
    return descs


def _seed_of(name: str) -> int:
    """A stable seed.  ``hash()`` on a str is randomised per process, and this
    runs in six of them."""
    return sum(name.encode()) & 0xFF


def _span(desc, side: str) -> int:
    if desc.kind == tp.STRIDED2D:
        return desc.rows * (desc.spitch if side == "src" else desc.dpitch)
    return desc.nbytes


def _rank_child(root: str, region_path: str, boot: str, group: str, rank: int,
                out_path: str, ready) -> None:
    """One of the six ranks, in its own process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    is_source = group == "P"
    row = xr.rank_row(group, rank)
    peer_row = xr.rank_row("D" if is_source else "P", rank)
    ops = FakeDeviceOps(root, rank=rank)
    verdict = {"row": row, "ok": False, "error": "", "lines": [],
               "mismatch": [], "zerofill": 0, "bump": 0, "waves": 0}
    sems = None
    try:
        with xr.XchgRegion.open(region_path, expect_boot=boot) as region:
            region.begin_flip(f"{boot}.1")
            region.bind(row)
            sems = tp.SemSet(boot)
            descs = _plan_for_double()
            for desc in descs:
                if not is_source:
                    continue
                if desc.kind == tp.ZEROFILL or int(desc.src_rank) != rank:
                    continue
                write(ops, int(desc.src_ptr) + int(desc.src_off),
                      pattern(_seed_of(desc.param_name), _span(desc, "src")))
            tp.register_region(region, ops, row, log=lambda _ln: None)
            ready.wait(30)
            for wave in range(DOUBLE_WAVES):
                if not is_source:
                    # RE-POISONED BEFORE EVERY WAVE, so wave 1's bytes can
                    # never be what makes wave 2's verification pass.  Safe
                    # without a barrier: a rank's device memory is written by
                    # that rank alone.
                    for desc in descs:
                        if int(desc.dst_rank) != rank:
                            continue
                        poison(ops, int(desc.dst_ptr) + int(desc.dst_off),
                               _span(desc, "dst"), seed=0x3C + wave)
                result = tp.run_leg(
                    region, sems, ops, row=row, rank=rank, device=0,
                    card_uuid=f"GPU-{rank}",
                    uuid_of_card=[f"GPU-{c}" for c in range(xr.N_CARDS)],
                    descs=descs, is_source=is_source,
                    oncard_mode=tp.ONCARD_MODE_IPC, peer_row=peer_row,
                    wave=wave,
                    log=lambda _ln: None,
                    vote_failure=lambda exc: region.write_gate_row(row, 0, False),
                    budget_s=40.0, slot_bytes=SLOT,
                )
                verdict["lines"] += list(result.lines)
                verdict["zerofill"] += result.zerofill_bytes
                verdict["waves"] += 1
                if not is_source:
                    verdict["mismatch"] += [f"w{wave}.{name}" for name in
                                            _verify(ops, root, descs, rank)]
            verdict["bump"] = ops._bump
            verdict["ok"] = not verdict["mismatch"]
    except BaseException as exc:  # noqa: BLE001 -- reported through the file
        verdict["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sems is not None:
            sems.close()
        ops.close()
        with open(out_path, "w") as fh:
            fh.write(repr(verdict))


def _verify(ops: FakeDeviceOps, root: str, descs, rank: int) -> list:
    """Byte identity against the SOURCE's own bytes, read back from its rank
    space -- never against a value this process recomputed.

    A test that compared the destination to its own idea of the payload would
    let a transport that moved nothing and an expectation that expected nothing
    agree.  ``ops.real`` resolves any rank's space, so the comparison reads the
    producer's memory directly.
    """
    bad = []
    for desc in descs:
        if int(desc.dst_rank) != rank:
            continue
        dst = int(desc.dst_ptr) + int(desc.dst_off)
        if desc.kind == tp.ZEROFILL:
            if read(ops, dst, desc.nbytes) != b"\x00" * desc.nbytes:
                bad.append(desc.param_name)
            continue
        src = int(desc.src_ptr) + int(desc.src_off)
        if desc.kind == tp.FLAT:
            if read(ops, src, desc.nbytes) != read(ops, dst, desc.nbytes):
                bad.append(desc.param_name)
            continue
        for r in range(desc.rows):
            if read(ops, src + r * desc.spitch, desc.run_bytes) != \
                    read(ops, dst + r * desc.dpitch, desc.run_bytes):
                bad.append(f"{desc.param_name}#row{r}")
                break
    return bad


def test_six_ranks_move_a_wave_and_every_byte_lands(tmp_path):
    """THE SLICE'S OWN SMOKE: six processes, both lanes, every class.

    Six ``multiprocessing`` children under ``CUDA_VISIBLE_DEVICES=""``, a real
    ``/dev/shm`` region, 24 real POSIX semaphores, the real batcher, the real
    slot state machine, the real leg -- TWICE, once per wave, with every
    destination re-poisoned in between.  The only substitution is the device
    adapter.

    The assertion is BYTE IDENTITY against the source's own bytes, and every
    destination was poisoned first.  The acceptance lines are counted by NAMED
    PAIR (R10): six directed pairs seen twice each, three diagonals seen twice
    each -- an aggregate would hide a pair that moved nothing.
    """
    boot = _fresh_boot()
    root = str(tmp_path / "dev")
    os.makedirs(root, exist_ok=True)
    region = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    path = region.path
    region.close()
    xr.create_semaphores(boot)
    ctx = mp.get_context("fork")
    ready = ctx.Barrier(xr.N_RANKS)
    procs, outs = [], []
    try:
        for group in ("P", "D"):
            for rank in range(xr.N_CARDS):
                out = str(tmp_path / f"v-{group}{rank}.txt")
                outs.append(out)
                proc = ctx.Process(
                    target=_rank_child,
                    args=(root, path, boot, group, rank, out, ready))
                proc.start()
                procs.append(proc)
        for proc in procs:
            proc.join(150)
        alive = [p for p in procs if p.is_alive()]
        assert not alive, f"{len(alive)} ranks never finished -- a lane deadlocked"
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        xr.unlink_semaphores(boot)

    verdicts = []
    for out in outs:
        with open(out) as fh:
            verdicts.append(eval(fh.read()))  # noqa: S307 -- our own repr
    for verdict in verdicts:
        assert verdict["error"] == "", f"row {verdict['row']}: {verdict['error']}"
    consumers = [v for v in verdicts if v["row"] >= xr.N_CARDS]
    assert len(consumers) == xr.N_CARDS
    for verdict in consumers:
        assert verdict["mismatch"] == [], verdict["mismatch"]
    assert all(v["waves"] == DOUBLE_WAVES for v in verdicts), \
        [v["waves"] for v in verdicts]
    assert sum(v["zerofill"] for v in consumers) == 256 * DOUBLE_WAVES
    # The on-card share is deeper than the bounce, so the producer's drain wait
    # is on the executed path rather than at its base case -- the gap that let
    # a deleted drain wait leave this file green.
    oncard_desc = [d for d in _plan_for_double()
                   if d.kind != tp.ZEROFILL and int(d.src_rank) == int(d.dst_rank)][0]
    assert len(tp.batch_descs([oncard_desc], SLOT)) > tp.ONCARD_SLOTS
    # The bounce is sized from run_leg's OWN slot size, not the module
    # default: a 4 KiB leg that allocated a 64 MiB bounce is exactly how
    # the half-applied knob was caught.
    sources = [v for v in verdicts if v["row"] < xr.N_CARDS]
    assert all(v["bump"] == DOUBLE_WAVES * tp.ONCARD_SLOTS * SLOT
               for v in sources), [v["bump"] for v in sources]
    assert all(v["bump"] == 0 for v in consumers), [v["bump"] for v in consumers]

    lines = [ln for v in verdicts for ln in v["lines"]]
    pairs = [ln for ln in lines if ln.startswith(tp.PAIR_LINE_PREFIX)]
    oncard = [ln for ln in lines if ln.startswith(tp.ONCARD_LINE_PREFIX)]
    assert len(pairs) == DOUBLE_WAVES * 2 * xr.N_PAIRS, pairs
    assert len(oncard) == DOUBLE_WAVES * 2 * xr.N_CARDS, oncard
    for src, dst in xr.CROSS_PAIRS:
        named = [ln for ln in pairs if f"src=GPU-{src} dst=GPU-{dst} " in ln]
        assert len(named) == DOUBLE_WAVES * 2, (src, dst, named)
        # NOT "bytes_mib != 0.00": these payloads are kilobytes and two
        # decimals of MiB cannot tell SMALL from NOTHING -- the double
        # printed exactly that for a lane that had just moved every byte
        # correctly.  The piece and hop counts can tell them apart, and they
        # are the fields that WOULD be zero on a lane that moved nothing.
        assert all(" pieces=0 " not in ln for ln in named), named
    for line in oncard:
        assert "mode=ipc" in line, line
        assert " batches=0 " not in line, line


# --- S6: the store-and-forward deposit -------------------------------------
#
# THE ORDERING THESE TESTS DRIVE is the one SECTION 1ai-S5c-fix documented and
# no test in this file had: the source hook runs, RETURNS, and only afterwards
# does the peer's leg exist (its resume is C14-fenced on a credit published
# inside the pause loop the source hook is upstream of).  The existing
# `test_the_degraded_lane_really_moves_the_bytes` runs both ends CONCURRENTLY
# in two threads, which is the shape that placement can never produce.


def _deposit_descs(prod, cons, *, batches: int, slot: int):
    """One on-card descriptor per slot, written and poisoned.  Returns descs."""
    descs = []
    for i in range(batches):
        payload = pattern(0x40 + i, slot)
        src, dst = dev_ptr(0, 0x10000 + i * slot), dev_ptr(0, 0x80000 + i * slot)
        write(prod, src, payload)
        poison(cons, dst, len(payload), seed=0xEE)
        descs.append(flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                               name=f"oncard{i}"))
    return descs


def test_a_deposit_outlives_its_leg_and_is_read_after_the_source_is_gone(
        tmp_path, region):
    """S6: the source deposits and RETURNS; the destination reads afterwards.

    THE CAN-FAIL THIS SLICE EXISTS FOR.  Nothing is concurrent here: the
    producer runs to completion on this thread, closes its own mapping (the
    leg's ``finally``) and only THEN is a consumer created.  On the shipping
    lane that sequence cannot work -- the producer's terminal drain waits for a
    consumer that does not exist yet and dies at the budget -- which is exactly
    what ``WEG2-XCHG-SHADOW-ONCARD-REFUSED`` six-per-flip was reporting.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    cons = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    root = os.path.dirname(os.path.dirname(region.path))
    try:
        batches = 5
        descs = _deposit_descs(prod, cons, batches=batches, slot=SLOT)
        assert len(tp.batch_descs(descs, SLOT)) == batches
        bounce = tp.HostBounce(prod, region.boot_nonce, 0, create=True,
                               slots=batches, slot_bytes=SLOT, shm_root=root)
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                            slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_ARMED)
        p_stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_HOST)
        tp.run_oncard_producer(region, prod, prod.create_stream(0), bounce,
                               row=0, peer_row=3, wave=WAVE, descs=descs,
                               stats=p_stats, budget_s=2.0, store_forward=True)
        # THE SOURCE LEG IS OVER.  It waited for nothing -- no per-batch drain
        # (every `seq - slots` is negative) and no terminal drain (skipped by
        # construction) -- and its mapping is gone.
        assert p_stats.batches == batches
        assert p_stats.drain_wait_outside_s == 0.0, p_stats.drain_wait_outside_s
        assert p_stats.drain_wait_s < 0.05, p_stats.drain_wait_s
        bounce.close()
        # ... and the bytes are still there, because `close` does not unlink.
        assert os.path.exists(bounce.path), bounce.path
        assert os.path.getsize(bounce.path) == batches * SLOT

        collected = tp.HostBounce(cons, region.boot_nonce, 0, create=False,
                                  slots=batches, slot_bytes=SLOT, shm_root=root)
        c_stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_HOST)
        tp.run_oncard_consumer(region, cons, cons.create_stream(0),
                               collected.ptr, row=3, peer_row=0, wave=WAVE,
                               descs=descs, stats=c_stats, slots=batches,
                               slot_bytes=SLOT, budget_s=2.0)
        collected.close()
        assert c_stats.batches == batches
        for i, desc in enumerate(descs):
            assert read(cons, desc.dst_ptr, desc.nbytes) == pattern(0x40 + i, SLOT)
    finally:
        prod.close()
        cons.close()


def test_a_deposit_with_fewer_slots_than_batches_is_refused_before_a_copy(
        tmp_path, region):
    """S6 W81: the danger direction, refused where it cannot be compensated.

    ``slots < batches`` under store-and-forward is SILENT: batch ``k`` and
    batch ``k + slots`` share a slot, the producer overwrites without waiting
    (that is what the deposit removed), and the destination -- reading in a
    later leg -- takes the later batch's bytes under the earlier batch's row
    with nothing disagreeing.  So the refusal is in the producer, before the
    first ``memcpy`` is issued, and not only in the planner.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    root = os.path.dirname(os.path.dirname(region.path))
    try:
        descs = _deposit_descs(prod, prod, batches=4, slot=SLOT)
        bounce = tp.HostBounce(prod, region.boot_nonce, 0, create=True,
                               slots=2, slot_bytes=SLOT, shm_root=root)
        issued = prod.issued
        with pytest.raises(tp.Weg2XchgDepositUnfundable) as caught:
            tp.run_oncard_producer(region, prod, prod.create_stream(0), bounce,
                                   row=0, peer_row=3, wave=WAVE, descs=descs,
                                   stats=tp.OnCardStats(0, "u0", "host"),
                                   budget_s=2.0, store_forward=True)
        bounce.close()
        assert "W81 Weg2XchgDepositUnfundable" in str(caught.value)
        assert "slots=2" in str(caught.value) and "batches=4" in str(caught.value)
        assert prod.issued == issued, "NO copy may be issued before the refusal"
    finally:
        prod.close()


def test_the_deposit_skips_the_terminal_drain_that_the_shipping_lane_needs(
        tmp_path, region):
    """S6: the same descriptors, with and without the deposit, one budget.

    The control is the point: without ``store_forward`` this producer blocks in
    ``drain-final`` for a consumer that will never come and dies at the budget
    naming that wait; with it, it returns.  A test that only ran the deposit
    would prove the new path works and nothing about what it removed.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    root = os.path.dirname(os.path.dirname(region.path))
    try:
        descs = _deposit_descs(prod, prod, batches=3, slot=SLOT)
        deposit = tp.HostBounce(prod, region.boot_nonce, 0, create=True,
                                slots=3, slot_bytes=SLOT, shm_root=root)
        stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_HOST)
        tp.run_oncard_producer(region, prod, prod.create_stream(0), deposit,
                               row=0, peer_row=3, wave=WAVE, descs=descs,
                               stats=stats, budget_s=0.2, store_forward=True)
        deposit.close()
        assert stats.batches == 3

        # THE CONTROL, same rows, same wave, same absent consumer.
        pipeline = tp.HostBounce(prod, region.boot_nonce, 1, create=True,
                                 slots=3, slot_bytes=SLOT, shm_root=root)
        with pytest.raises(tp.Weg2XchgGateTimeout) as caught:
            tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                   pipeline, row=0, peer_row=3, wave=WAVE,
                                   descs=descs,
                                   stats=tp.OnCardStats(0, "u0", "host"),
                                   budget_s=0.2, store_forward=False)
        pipeline.close()
        assert "what=drain-final" in str(caught.value), caught.value
    finally:
        prod.close()


def test_the_slot_count_is_derived_from_the_batches_and_prints_its_provenance():
    """S6: ``slots`` stops being an echoed constant and says where it came from.

    Every product path ran the lane at ``tp.ONCARD_SLOTS = 2`` because
    ``run_leg_hook`` had no way to pass anything else -- a hand number one seam
    below a planner that already knew the batch count.  Under the deposit the
    count IS the batch count, and the plan is its one producer, so the W68
    cross-check between the two co-located processes still compares two
    readings of one derivation.
    """
    default = tp.plan_oncard_slot_bytes(200 * xr.MIB)
    assert default.slots == tp.ONCARD_SLOTS and default.slots_source == "caller"

    plan = tp.plan_oncard_slot_bytes(200 * xr.MIB, store_forward=True)
    assert plan.slots == plan.batches, (plan.slots, plan.batches)
    assert plan.slots_source == "store-forward-batches"
    assert plan.deposit_bytes == plan.slots * plan.slot_bytes
    for token in ("oncard_slots_source=store-forward-batches",
                  f"oncard_slots={plan.slots}",
                  f"oncard_deposit_mib={plan.deposit_bytes / xr.MIB:.0f}"):
        assert token in plan.tokens(), (token, plan.tokens())
    # THE SIZE IS RAISED UNTIL THE COUNT FITS THE ROW AREA, never clamped: a
    # clamp would be the `slots < batches` overwrite arrived at by arithmetic.
    big = tp.plan_oncard_slot_bytes(tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES,
                                    store_forward=True)
    assert big.slots == big.batches <= tp.ONCARD_SLOTS_MAX
    over = tp.plan_oncard_slot_bytes(
        4 * tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES_MAX, store_forward=True)
    assert over.slots == over.batches > tp.ONCARD_SLOTS_MAX


def test_the_three_deposit_refusals_are_named_apart():
    """S6: three causes, three levers, three words -- never one 'refused'."""
    budget = tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES
    ok = dict(batches=2, slots=2, slot_bytes=tp.ONCARD_SLOT_BYTES,
              budget_bytes=budget, mode=tp.ONCARD_MODE_HOST)
    assert tp.deposit_refusal_reason(**ok) == ""
    assert tp.deposit_refusal_reason(**{**ok, "mode": tp.ONCARD_MODE_IPC}) == \
        tp.DEPOSIT_REASON_IPC
    assert tp.deposit_refusal_reason(
        **{**ok, "batches": tp.ONCARD_SLOTS_MAX + 1,
           "slots": tp.ONCARD_SLOTS_MAX + 1}) == tp.DEPOSIT_REASON_BATCHES
    assert tp.deposit_refusal_reason(**{**ok, "slots": 1}) == \
        tp.DEPOSIT_REASON_BATCHES
    # An unfunded deposit and an UNREAD budget refuse the same way: an absent
    # measurement never becomes a quiet zero.
    assert tp.deposit_refusal_reason(**{**ok, "budget_bytes": 0}) == \
        tp.DEPOSIT_REASON_UNFUNDED
    assert tp.deposit_refusal_reason(
        **{**ok, "slot_bytes": budget}) == tp.DEPOSIT_REASON_UNFUNDED


def test_run_leg_refuses_a_deposit_on_the_exported_arm(tmp_path, region, boot):
    """S6: an exported VRAM bounce cannot outlive the leg that exported it.

    Refused BEFORE a thread exists, like the two slot knobs beside it: the
    exporter's ``cudaFree`` runs in this leg's own unwind, so a destination
    reading in a later leg would map freed VRAM -- and half-honouring the ask
    (deposit, then free with the leg) is the use-after-free the S4 review
    already found once.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        with pytest.raises(tp.Weg2XchgDepositUnfundable) as caught:
            tp.run_leg(region, sems, ops, row=0, rank=0, device=0,
                       card_uuid="u0", uuid_of_card=("u0", "u1", "u2"),
                       descs=[], is_source=True,
                       oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
                       log=lambda _s: None, vote_failure=lambda _e: None,
                       slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                       oncard_store_forward=True)
        assert tp.DEPOSIT_REASON_IPC in str(caught.value)
        assert "W81 Weg2XchgDepositUnfundable" in str(caught.value)
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)


# ===========================================================================
# S6 FIX -- the five must_fix of the S6 refuter.  Each test names the defect
# it closes and carries the control that would have passed before the fix.
# ===========================================================================


def test_the_host_arm_prices_the_copy_and_the_ipc_arm_must_not():
    """S6 must_fix 1: the deposit's cost gate priced no copy at all.

    ``ONCARD_PER_BATCH_MS`` was measured on the ``ipc`` arm, where the copy is
    device-to-device and disappears inside the handshake.  Pricing a ``host``
    deposit with it asserts 512 MiB in 2.91 ms = 171.8 GiB/s over a link this
    rig does not have, and S6 made that load-bearing: a deposit is host-only by
    construction, so every batch it prices is a real unoverlapped PCIe copy.

    THE FALSE GREEN, EXACTLY: one card's deposit at the geometry's ceiling
    clears the shadow's 60 ms hop bound at the old model and does not at the
    new one.  A mutant that drops the copy term fails on the second assert.
    """
    assert tp.oncard_copy_gbps(tp.ONCARD_MODE_HOST) == tp.ONCARD_HOST_COPY_GBPS
    assert tp.oncard_copy_gbps(tp.ONCARD_MODE_IPC) == 0.0

    ceiling = tp.ONCARD_DEPOSIT_BYTES_MAX
    bound = 3.0 * tp.ONCARD_HOP_BUDGET_MS          # the shadow's own bound
    old = tp.plan_oncard_slot_bytes(ceiling, store_forward=True)
    new = tp.plan_oncard_slot_bytes(ceiling, store_forward=True,
                                    copy_gbps=tp.ONCARD_HOST_COPY_GBPS)
    assert old.copy_ms == 0.0 and old.hop_ms < bound, old.model()
    assert new.hop_ms > bound, new.model()
    # ... and the term that made the difference is the BYTES, by two orders.
    assert new.copy_ms > 100 * (new.batches * new.per_batch_ms), new.model()
    assert new.copy_ms == pytest.approx(
        (ceiling / (tp.ONCARD_HOST_COPY_GBPS * 1e9)) * 1e3)
    # The model PRINTS both terms, because they have different levers: too
    # many batches is a slot-size question, too many bytes is not.
    assert f"{tp.ONCARD_HOST_COPY_GBPS:g} GB/s" in new.model(), new.model()
    assert "not priced" in old.model(), old.model()
    for token in (f"oncard_copy_gbps={tp.ONCARD_HOST_COPY_GBPS:g}",
                  f"oncard_copy_ms={new.copy_ms:.1f}"):
        assert token in new.tokens(), (token, new.tokens())


def test_the_copy_is_paid_out_of_the_hop_budget_before_the_batches_are():
    """S6 must_fix 1, the arithmetic half: bytes first, batches from the rest.

    A slot size chosen from the WHOLE budget spends the copy's share twice.
    When the copy alone exceeds the budget the plan takes the largest slot the
    ceiling allows -- the fewest batches the geometry can cut -- and says
    ``fits=False`` rather than clamping something to make a green.
    """
    total = 8 * tp.ONCARD_SLOT_BYTES_MAX
    free = tp.plan_oncard_slot_bytes(total, budget_ms=20.0)
    paid = tp.plan_oncard_slot_bytes(total, budget_ms=20.0,
                                     copy_gbps=tp.ONCARD_HOST_COPY_GBPS)
    assert paid.copy_ms > 20.0, paid.model()
    assert paid.slot_bytes == tp.ONCARD_SLOT_BYTES_MAX
    assert paid.fits is False and paid.hop_ms > paid.budget_ms
    # THE CAN-FAIL: the same call with no copy term fits, and its slot is the
    # smaller one the un-spent budget bought.
    assert free.slot_bytes <= paid.slot_bytes
    # A zero rate is the ipc arm and reproduces the pre-S6 arithmetic exactly.
    assert tp.plan_oncard_slot_bytes(total, budget_ms=20.0, copy_gbps=0.0) == free


def test_the_deposit_is_declared_to_the_pinned_host_owner_before_it_is_pinned(
        tmp_path, region):
    """S6 must_fix 2: the deposit pinned host RAM outside its declared owner.

    ``pinned_host_budget`` is the #550 single owner of "may this PINNED host
    buffer be allocated?", and the joint check HiCache and kv-session-offload
    pass through sums ``registered_posts()``.  The bounce called
    ``cudaHostRegister`` with no post at all, so that sum was short by up to
    the whole deposit -- two ledgers for one payload.  The #1269 ledger term
    S6 added answers a different question (the launch-time reap bound); this
    one answers at the moment of allocation.
    """
    from sglang.srt.mem_cache import pinned_host_budget as php

    root = os.path.dirname(os.path.dirname(region.path))
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    php.clear_registered_posts()
    try:
        bounce = tp.HostBounce(ops, region.boot_nonce, 0, create=True,
                               slots=4, slot_bytes=SLOT, shm_root=root)
        posts = [p for p in php.registered_posts() if bounce.path in p.name]
        assert len(posts) == 1, php.registered_posts()
        assert posts[0].nbytes == 4 * SLOT
        assert posts[0].flag == tp.HostBounce.POST_FLAG
        bounce.close()
        # ... and RELEASED, or the next admission is charged for bytes nobody
        # holds -- the mirror of the under-charge this closes.
        assert not [p for p in php.registered_posts() if bounce.path in p.name]

        # DECLARED BEFORE ALLOCATED (#729's ordering), proven by refusing at
        # the declaration: no file may exist afterwards.
        path = tp.oncard_host_path(region.boot_nonce, 7, root)
        assert not os.path.exists(path)

        def _refuse(name, flag, nbytes, **kw):
            raise RuntimeError("over-committed")

        real = tp.check_and_register_pinned_post
        tp.check_and_register_pinned_post = _refuse
        try:
            with pytest.raises(RuntimeError):
                tp.HostBounce(ops, region.boot_nonce, 7, create=True,
                              slots=2, slot_bytes=SLOT, shm_root=root)
        finally:
            tp.check_and_register_pinned_post = real
        assert not os.path.exists(path), "the post is declared BEFORE the map"

        # ... and a post whose allocation then fails is undone (#729).
        real_makedirs = os.makedirs
        os.makedirs = lambda *a, **k: (_ for _ in ()).throw(OSError("no dir"))
        try:
            with pytest.raises(OSError):
                tp.HostBounce(ops, region.boot_nonce, 8, create=True,
                              slots=2, slot_bytes=SLOT, shm_root=root)
        finally:
            os.makedirs = real_makedirs
        leftover = [p for p in php.registered_posts() if "oncard-8.bin" in p.name]
        assert leftover == [], leftover
    finally:
        php.clear_registered_posts()
        ops.close()


def test_the_charged_deposit_is_the_geometry_the_planner_can_actually_derive():
    """S6 must_fix 3: the charge read the slot FLOOR beside the count CEILING.

    ``ONCARD_SLOTS_MAX x ONCARD_SLOT_BYTES`` is 256 MiB; the shape's maximum is
    ``ONCARD_SLOTS_MAX x ONCARD_SLOT_BYTES_MAX`` = 1024 MiB, and
    ``plan_oncard_slot_bytes`` clamps to the latter.  The consequence was not
    an overspend but a systematic FALSE REFUSAL, because
    ``deposit_refusal_reason`` grades against the same number: every per-card
    diagonal in the 256 MiB..1 GiB band -- the band the geometry exists for --
    refused ``ledger-cannot-fund-deposit``.
    """

    assert tp.ONCARD_DEPOSIT_BYTES_MAX == \
        tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES_MAX
    # AMENDMENT 5: the ledger no longer charges this ceiling. The deposit's
    # own SHAPE maximum still exists (it is the transport's, and the refusal
    # below still grades a plan against a budget), but the CHARGE is
    # `xchg_bounce.staging_bytes_per_card(published slot)`, and the two are
    # deliberately different numbers now: 8 x slot vs SLOTS_PER_PAIR x slot.
    assert xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX) == xb.SLOTS_PER_PAIR * tp.ONCARD_SLOT_BYTES_MAX
    assert xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX) < tp.ONCARD_DEPOSIT_BYTES_MAX
    assert tp.ONCARD_DEPOSIT_BYTES_MAX == \
        4 * tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES, "the 4x understatement"

    # A plan in the band the fix restores: funded at the real ceiling, refused
    # at the old floor-derived one.
    plan = tp.plan_oncard_slot_bytes(4 * tp.ONCARD_SLOT_BYTES_MAX,
                                     store_forward=True)
    graded = dict(batches=plan.batches, slots=plan.slots,
                  slot_bytes=plan.slot_bytes, mode=tp.ONCARD_MODE_HOST)
    assert plan.deposit_bytes > tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES
    assert tp.deposit_refusal_reason(
        budget_bytes=tp.ONCARD_DEPOSIT_BYTES_MAX, **graded) == ""
    assert tp.deposit_refusal_reason(
        budget_bytes=tp.ONCARD_SLOTS_MAX * tp.ONCARD_SLOT_BYTES,
        **graded) == tp.DEPOSIT_REASON_UNFUNDED

    # The card count is DERIVED from the region's own rank layout, not typed.
    # The GROUP-WIDE charge is the term's staging, priced once by the
    # launcher: N_CARDS x SLOTS_PER_PAIR x published slot.
    assert xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX) == xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX)
    assert xr.N_CARDS == xr.N_RANKS // 2


def test_the_two_sides_ring_depths_are_compared_and_not_assumed(
        tmp_path, region, boot):
    """S6 must_fix 5: ``slots`` crossed the two processes with no cross-check.

    Only ``slot_bytes`` was compared.  With equal slot sizes and unequal
    depths the consumer polls ``seq % slots_dst`` while the producer wrote
    ``seq % slots_src``: every wait misses, and the destination burns its whole
    budget in a W69 whose denominator names an ABSENT peer -- never a ring
    sized differently on the two sides.  The ARMED row's ``nbytes`` word is
    written on a row where no batch exists, so it was free.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    payload = pattern(0x11, SLOT)
    src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x80000)
    write(ops, src, payload)
    descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                       name="oncard")]
    try:
        # The producer's own row now CARRIES the depth it armed.
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1,
                            nbytes=4, slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_ARMED)
        got = tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, 0)
        assert got["bytes"] == 4 and got["slot_bytes"] == SLOT

        with pytest.raises(xr.Weg2XchgPlanDisagree) as caught:
            tp.run_leg(region, sems, ops, row=3, rank=0, device=0,
                       card_uuid="u0", uuid_of_card=("u0", "u1", "u2"),
                       descs=descs, is_source=False,
                       oncard_mode=tp.ONCARD_MODE_HOST, peer_row=0, wave=WAVE,
                       log=lambda _s: None, vote_failure=lambda _e: None,
                       budget_s=1.0, slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                       oncard_slots=2)
        message = str(caught.value)
        assert "W68 Weg2XchgPlanDisagree" in message
        assert "slots=4" in message and "derived 2" in message
        assert "different depths" in message

        # THE CAN-FAIL CONTROL: agreeing depths do NOT raise W68.  This rank
        # then dies on the missing bounce file, which is a different failure
        # and proves the check above discriminated rather than always fired.
        with pytest.raises(BaseException) as other:
            tp.run_leg(region, sems, ops, row=3, rank=0, device=0,
                       card_uuid="u0", uuid_of_card=("u0", "u1", "u2"),
                       descs=descs, is_source=False,
                       oncard_mode=tp.ONCARD_MODE_HOST, peer_row=0, wave=WAVE,
                       log=lambda _s: None, vote_failure=lambda _e: None,
                       budget_s=1.0, slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                       oncard_slots=4)
        assert not isinstance(other.value, xr.Weg2XchgPlanDisagree), other.value
        assert isinstance(other.value, OSError), other.value
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)


def test_a_lane_that_deposits_nothing_prints_no_deposit(tmp_path):
    """S6 refuter finding 8: ``oncard_deposit_mib`` on a non-depositing lane.

    The field's name means "pinned host bytes held ACROSS the flip".  On the
    drainable lane the slots are a pipelined double buffer allocated and freed
    inside the leg, so printing them made a sum over acceptance lines count
    host bytes that were never held.
    """
    drain = tp.plan_oncard_slot_bytes(200 * xr.MIB)
    keep = tp.plan_oncard_slot_bytes(200 * xr.MIB, store_forward=True)
    assert drain.slots > 0 and drain.deposit_bytes == 0, drain.tokens()
    assert keep.deposit_bytes == keep.slots * keep.slot_bytes
    assert "oncard_deposit_mib=0 " in drain.tokens() + " "


def test_the_source_publishes_its_ring_depth_in_the_arming_row(
        tmp_path, region, boot):
    """S6 must_fix 5, the PRODUCER half, deterministic.

    The consumer's cross-check is only as good as what the producer writes, and
    the co-located pair's concurrent tests catch a producer that writes 0 only
    when the consumer happens to read the row before batch 0 overwrites it.
    This drives ``run_leg`` as the SOURCE and reads the row back, so the
    publication is proven by arithmetic rather than by a race.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    payload = pattern(0x21, SLOT)
    src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x80000)
    write(ops, src, payload)
    descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                       name="oncard")]
    try:
        tp.run_leg(region, sems, ops, row=0, rank=0, device=0,
                   card_uuid="u0", uuid_of_card=("u0", "u1", "u2"),
                   descs=descs, is_source=True,
                   oncard_mode=tp.ONCARD_MODE_HOST, peer_row=3, wave=WAVE,
                   log=lambda _s: None, vote_failure=lambda _e: None,
                   budget_s=2.0, slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                   oncard_slots=5, oncard_store_forward=True)
        armed = tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, 0)
        # The row may have advanced to the batch it then wrote; the ARMING
        # row's own word is what this pins, so read it while it is still one.
        if armed["state"] == tp.ONCARD_STATE_ARMED:
            assert armed["seq"] == -1
            assert armed["bytes"] == 5, armed
        assert armed["slot_bytes"] == SLOT, armed
        # ... and the write itself, at the one place it happens.
        source = inspect.getsource(tp.run_leg)
        assert "seq=-1, nbytes=diag_slots," in source, \
            "the arming row must carry the ring depth, not a zero"
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
