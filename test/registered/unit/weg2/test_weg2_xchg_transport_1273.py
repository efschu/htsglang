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
two on-card hops, the five threads.  Six real processes run it.  What the fake
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

#: Small enough that batching, slot alternation and the double buffer are all
#: exercised by a few kilobytes of payload.
SLOT = 4096
FAKE_DEV_BYTES = 8 << 20

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

FAKE_DEV_BASE = 1 << 46
FAKE_DEV_SPAN = 1 << 32
FAKE_IMPORT_BIAS = 1 << 45


def dev_ptr(rank: int, off: int) -> int:
    return FAKE_DEV_BASE + int(rank) * FAKE_DEV_SPAN + int(off)


class FakeDeviceOps(tp.DeviceOps):
    """:class:`tp.DeviceOps` over bytes.  Cross-process by construction.

    "Device memory" is one file per rank under ``root/dev-<rank>.bin``; a
    device pointer is ``FAKE_DEV_BASE + rank * SPAN + offset``, so which rank a
    copy landed on is recoverable from the address alone.  Host pointers are
    REAL addresses (the mapped shm region), so the staging layout under test is
    the product's own and not a model of it.
    """

    name = "fake"

    def __init__(self, root: str, rank: int, *, fail_ipc_open: bool = False,
                 fail_host_register: bool = False, drop_last_copy: bool = False,
                 stream_flags: int = tp.CUDA_STREAM_DEFAULT):
        self.root = root
        self.rank = int(rank)
        self.fail_ipc_open = fail_ipc_open
        self.fail_host_register = fail_host_register
        self.drop_last_copy = drop_last_copy
        self._stream_flags = int(stream_flags)
        self._maps: dict = {}
        self._bump = 0
        self._streams: dict = {}
        self._next_stream = 1
        self._lock = threading.Lock()
        self.registered: list = []
        self.issued = 0
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

    def real(self, ptr: int) -> int:
        """Resolve a fake pointer to a real address.  Host pointers pass through."""
        raw = int(ptr) & ~FAKE_IMPORT_BIAS
        if raw < FAKE_DEV_BASE:
            return int(ptr)
        rank = (raw - FAKE_DEV_BASE) // FAKE_DEV_SPAN
        off = (raw - FAKE_DEV_BASE) % FAKE_DEV_SPAN
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
        return dev_ptr(self.rank, off)

    def raw_free(self, ptr: int) -> None:
        return None

    def ipc_get_handle(self, ptr: int) -> bytes:
        raw = int(ptr) & ~FAKE_IMPORT_BIAS
        rank = (raw - FAKE_DEV_BASE) // FAKE_DEV_SPAN
        off = (raw - FAKE_DEV_BASE) % FAKE_DEV_SPAN
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
        return None

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
    """The producer posts ``bytes_filled = plan - 1``; the consumer raises W54
    and issues NO copy.

    This is #802's physical root, which #802 itself explicitly did not close.
    The two halves of the assertion carry equal weight: the NAME must be in the
    message so a grep over a boot log finds it, and ``ops.issued`` must not
    move, so the refusal is PROVEN to precede the copy rather than to follow
    it.  A W54 raised after the copy is a log line, not a guard.
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
    assert ops.issued == before, "W54 must precede the first copy, not follow it"
    assert read(ops, dst, len(payload)) != payload


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
    assert "W53 Weg2XchgGateTimeout" in str(excinfo.value)
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
    """A failing ``cudaIpcOpenMemHandle`` arms **W56** at LAUNCH and the lane
    degrades to the staging region with a named line.

    Four things are asserted, and the third and fourth are the ones that
    matter: the code and the degrade target; that the decision is taken at the
    ARM (``probe`` runs exactly once, so nothing per-flip and nothing per-lane
    can change it later -- R2-1); and that an explicit request for the degrade
    is not overridden by a green probe.
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


def test_the_degraded_lane_really_moves_the_bytes(tmp_path, region):
    """W56 names a degrade; the degrade has to work.

    A named fallback that was never executed is the
    desk-written-never-executed class.  The ``host`` arm runs the same batcher,
    the same rows and the same W54 comparison over a per-card shm bounce
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
                            state=tp.ONCARD_STATE_ARMED)
        c_bounce = tp.HostBounce(cons, region.boot_nonce, 0, create=False,
                                 slot_bytes=SLOT, shm_root=os.path.dirname(
                                     os.path.dirname(region.path)))
        assert c_bounce.ptr != p_bounce.ptr, "two mappings, two addresses"
        errors: list = []

        def produce():
            try:
                tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                       p_bounce, row=0, peer_row=3, descs=descs,
                                       stats=tp.OnCardStats(0, "u0", "host"),
                                       budget_s=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_HOST)
        tp.run_oncard_consumer(region, cons, cons.create_stream(0), c_bounce.ptr,
                               row=3, peer_row=0, descs=descs, stats=stats,
                               slot_bytes=SLOT, budget_s=10.0)
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

    If this were false, EVERY boot would fire W54 and the refusal would be
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
    assert "W52 Weg2XchgPlanDisagree" in str(excinfo.value)
    assert "run_bytes" in str(excinfo.value)


def test_an_unknown_copy_kind_is_refused_not_treated_as_flat():
    desc = flat_desc(0, 1, 64, src_ptr=dev_ptr(0, 0), dst_ptr=dev_ptr(1, 0))
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        tp.batch_descs([desc.replace(kind="RESHAPE2D")], SLOT)
    assert "RESHAPE2D" in str(excinfo.value)


def test_a_batch_never_exceeds_the_slot_it_will_be_written_into():
    """The one arithmetic invariant the whole handshake rests on.

    ``publish`` refuses a ``bytes_filled`` over ``SLOT_BYTES``, so a batcher
    that over-packed would turn every flip into a W52 -- but only after the
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
                            state=tp.ONCARD_STATE_ARMED)

        peer = cons.ipc_open_handle(tp.read_ipc_handle(region, 0))
        assert peer != bounce.ptr, "an imported IPC pointer is a DIFFERENT address"

        pstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        cstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        errors: list = []

        def produce():
            try:
                tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                       bounce, row=0, peer_row=3, descs=descs,
                                       stats=pstats, budget_s=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        tp.run_oncard_consumer(region, cons, cons.create_stream(0), peer,
                               row=3, peer_row=0, descs=descs, stats=cstats,
                               slots=tp.ONCARD_SLOTS, slot_bytes=SLOT,
                               budget_s=10.0)
        thread.join(30)
        assert not errors, errors
        assert read(cons, dst, len(payload)) == payload
        assert pstats.hops == cstats.hops > 1, "the double buffer really cycled"
        bounce.close()
    finally:
        prod.close()
        cons.close()


def test_oncard_consumer_refuses_a_short_hop(region, tmp_path):
    """W54 on the diagonal too -- one refusal, both lanes.

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
                            state=tp.ONCARD_STATE_READY)
        before = ops.issued
        with pytest.raises(tp.Weg2XchgShortPiece) as excinfo:
            tp.run_oncard_consumer(region, ops, ops.create_stream(0), bounce.ptr,
                                   row=3, peer_row=0, descs=descs,
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
                        state=tp.ONCARD_STATE_READY)
    assert tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 5, 1.0,
                            what="fill", row=3)["bytes"] == 99
    region.begin_flip(f"{boot}.2")
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 5, 0.2,
                         what="fill", row=3)
    assert "denominator" in str(excinfo.value)
    assert "epoch_hash" in str(excinfo.value)
    assert "alive_in_proc" in str(excinfo.value)


def test_a_torn_oncard_row_is_not_a_signal(region):
    """Half a row must not read as a whole one -- the same seal law as S3's
    gate rows, in the area S4 owns."""
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 1, seq=2, nbytes=64,
                        state=tp.ONCARD_STATE_READY)
    assert tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 1)["sealed"] == 1
    view = region.dir_view()
    off = tp._oncard_row_off(tp.DIR_ONCARD_PROD_OFF, 1)
    ctypes.memmove(ctypes.addressof(view) + off + 8, struct.pack("<Q", 999), 8)
    assert tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 1)["sealed"] == 0
    with pytest.raises(xr.Weg2XchgGateTimeout):
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 1, 2, 0.2,
                         what="fill", row=4)


def test_an_armed_row_is_the_handle_signal_and_not_a_batch(region):
    """``ARMED``/``seq=-1`` means "the bounce exists", never "batch 0 is ready".

    Folding the two would make batch 0's wait also the handle's wait, and the
    consumer would then open a buffer it had never been told was there --
    or, worse, treat the arming row as a filled batch of zero bytes and raise
    W54 against a producer that had done nothing wrong.
    """
    tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                        state=tp.ONCARD_STATE_ARMED)
    tp._await_handle(region, 0, 1.0, row=3)          # returns at once
    with pytest.raises(xr.Weg2XchgGateTimeout):      # but is not batch 0
        tp._await_oncard(region, tp.DIR_ONCARD_PROD_OFF, 0, 0, 0.2,
                         what="fill", row=3)


def test_await_handle_names_a_source_that_never_armed(region):
    with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
        tp._await_handle(region, 2, 0.2, row=5)
    message = str(excinfo.value)
    assert "what=handle" in message
    assert "peer_row=2" in message
    assert "never published a bounce handle" in message


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
    ]
    cursor = 0
    for name, off, size in areas:
        assert off == cursor, f"{name} does not abut its predecessor"
        cursor = off + size
    assert cursor == tp.DIR_USED_BYTES <= tp.DIR_CAPACITY
    assert tp.DIR_CAPACITY == xr.DATA_OFF - xr.DIR_OFF
    assert tp.ONCARD_ROW_STRUCT.size + 8 <= tp.DIR_ONCARD_ROW_BYTES
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
    """
    sig = inspect.signature(tp.run_leg)
    required = {n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty}
    assert required == {
        "region", "sems", "ops", "row", "rank", "device", "card_uuid",
        "uuid_of_card", "descs", "is_source", "oncard_mode", "peer_row",
        "log", "vote_failure",
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
                            bytes_moved=2 * xr.MIB, elapsed_s=0.013)
    for token in ("WEG2-XCHG-ONCARD", "card=GPU-card", "mode=ipc",
                  "bytes_mib=2.00", "hop_ms=13.000"):
        assert token in oncard.line(), token


def test_every_w_code_this_slice_raises_is_free_and_named_once():
    """W54 and W56 are what spec section 7 assigns to S4, and the branch census
    at 2ee844f7b8 leaves exactly those two free below W60 (W51/52/53/55/58/60
    are held by S1/S2/S3/S7; W57 is S6's and W59 is S5's).

    ``test_weg2_wcode_uniqueness_1263`` is the authority and walks the whole
    tree; this asserts the local half, so a collision introduced here is named
    here rather than three files away.
    """
    assert tp.SHORT_PIECE_MARKER == "W54 Weg2XchgShortPiece"
    assert tp.ONCARD_UNAVAILABLE_MARKER == "W56 Weg2XchgOnCardUnavailable"
    found = set(re.findall(r"\b(W\d{1,2}[a-z]?)\s+(Weg2[A-Za-z0-9_]+)",
                           inspect.getsource(tp)))
    assert found == {
        ("W52", "Weg2XchgPlanDisagree"),
        ("W53", "Weg2XchgGateTimeout"),
        ("W54", "Weg2XchgShortPiece"),
        ("W56", "Weg2XchgOnCardUnavailable"),
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


def _cross_off(src: int, dst: int) -> int:
    return CROSS_BASE + (src * xr.N_CARDS + dst) * CROSS_STRIDE


def _plan_for_double():
    """Three cards, two cross classes per directed pair, plus the diagonal.

    Deliberately mixed -- FLAT and STRIDED2D on every cross pair, an on-card
    share on every card, one ZEROFILL -- because the failure this double exists
    to catch is a lane that works alone and deadlocks beside the others.
    """
    descs = []
    for src in range(xr.N_CARDS):
        for dst in range(xr.N_CARDS):
            if src == dst:
                descs.append(flat_desc(
                    src, dst, 5000, src_off=ONCARD_SRC, dst_off=ONCARD_DST,
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
               "mismatch": [], "zerofill": 0}
    sems = None
    try:
        with xr.XchgRegion.open(region_path, expect_boot=boot) as region:
            region.begin_flip(f"{boot}.1")
            region.bind(row)
            sems = tp.SemSet(boot)
            descs = _plan_for_double()
            for desc in descs:
                if is_source:
                    if desc.kind == tp.ZEROFILL or int(desc.src_rank) != rank:
                        continue
                    write(ops, int(desc.src_ptr) + int(desc.src_off),
                          pattern(_seed_of(desc.param_name), _span(desc, "src")))
                else:
                    if int(desc.dst_rank) != rank:
                        continue
                    poison(ops, int(desc.dst_ptr) + int(desc.dst_off),
                           _span(desc, "dst"), seed=0x3C)
            tp.register_region(region, ops, row, log=lambda _ln: None)
            ready.wait(30)
            result = tp.run_leg(
                region, sems, ops, row=row, rank=rank, device=0,
                card_uuid=f"GPU-{rank}",
                uuid_of_card=[f"GPU-{c}" for c in range(xr.N_CARDS)],
                descs=descs, is_source=is_source,
                oncard_mode=tp.ONCARD_MODE_IPC, peer_row=peer_row,
                log=lambda _ln: None,
                vote_failure=lambda exc: region.write_gate_row(row, 0, False),
                budget_s=40.0, slot_bytes=SLOT,
            )
            verdict["lines"] = list(result.lines)
            verdict["zerofill"] = result.zerofill_bytes
            if not is_source:
                verdict["mismatch"] = _verify(ops, root, descs, rank)
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
    slot state machine, the real five-thread leg.  The only substitution is the
    device adapter.

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
    assert sum(v["zerofill"] for v in consumers) == 256

    lines = [ln for v in verdicts for ln in v["lines"]]
    pairs = [ln for ln in lines if ln.startswith(tp.PAIR_LINE_PREFIX)]
    oncard = [ln for ln in lines if ln.startswith(tp.ONCARD_LINE_PREFIX)]
    assert len(pairs) == 2 * xr.N_PAIRS, pairs
    assert len(oncard) == 2 * xr.N_CARDS, oncard
    for src, dst in xr.CROSS_PAIRS:
        named = [ln for ln in pairs if f"src=GPU-{src} dst=GPU-{dst} " in ln]
        assert len(named) == 2, (src, dst, named)
        assert all("bytes_mib=0.00" not in ln for ln in named), named
    for line in oncard:
        assert "mode=ipc" in line and "bytes_mib=0.00" not in line
