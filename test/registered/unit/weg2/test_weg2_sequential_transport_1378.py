# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn44: DER SEQUENTIELLE EINHEITEN-TRANSPORT + BEIDE ZEUGEN.

DAS DESIGN: EIN Host-Puffer je LANE, die Einheiten = die Stuecke des Lane-
Desc-Listen-Joins, der Deposit kopiert das Stueck in sein Fenster und postet
EIN Signal, der Collect wartet auf das Signal, liest sein Fenster, kopiert in
sein Ziel.  Sequenziell, ohne dass der Deposit auf den Collector wartet (die
#1374-Vertragsform: full ist COUNTING, der einzige Wait ist der je Tag).

DIE ZWEI ZEUGEN:
* MUTANT 1 (Transport, sha256-Fenster-Digest): die Einheiten rotiert ->
  der Digest stirbt (die falschen Bytes gelesen).
* MUTANT 2 (Platzierung, der Ziel-Digest): die Zieladressen vertauscht ->
  der Puffer-Digest laeuft GRUENFAELSCH, aber die Bytes stehen am Ziel an
  der falschen Stelle.  Stirbt am Ziel-Digest (dst_digest_fn).

xsn53-KORREKTUR: die Tests fahren jetzt DESC-Liste + ``slot_bytes`` ueber
``run_sequential_units`` (die Produktform), nicht mehr die fruehere
Einheiten-Tupel-Form -- und der Puffer ist das GETEILTE tmpfs-mmap, sodass
die Digests ueber dieselben Bytes laufen, die der Deposit geschrieben hat.
Die fruehere Form gruentgte auf einem PRIVATEN Puffer, in dem beide Seiten
denselben Null-Digest bekamen (der int/int-Zweig des Fake-Ops kopierte
nichts) -- gruen bei Null transportierter Bytes.
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402


class _MemOps:
    """The same address-honest ops the xsn53 identity file uses, inlined so
    this module stays importable without a sibling on sys.path."""

    def __init__(self):
        self.vram = {}
        self.vram_real = {}
        self.vram_size = {}

    def hook(self, addr, nbytes):
        import ctypes
        buf = ctypes.create_string_buffer(nbytes)
        # KEEP the object alive: only the address would let the allocator hand
        # the same block out twice, which is a corruption the digest then
        # cannot attribute.
        self.vram[addr] = buf
        self.vram_real[addr] = ctypes.addressof(buf)
        self.vram_size[addr] = nbytes

    def _real(self, addr):
        for fake, real in self.vram_real.items():
            if fake <= addr < fake + self.vram_size[fake]:
                return real + (addr - fake)
        return addr  # the shared mmap itself

    def write(self, addr, data):
        import ctypes
        ctypes.memmove(self._real(addr), bytes(data), len(data))

    def read(self, addr, nbytes):
        import ctypes
        return ctypes.string_at(self._real(addr), nbytes)

    def digest(self, addr, nbytes):
        import hashlib
        return hashlib.sha256(self.read(addr, nbytes)).hexdigest()[:16]

    def memcpy_async(self, dst, src, nbytes, stream):
        import ctypes
        ctypes.memmove(self._real(dst), self._real(src), nbytes)

    def memcpy2d_async(self, dst, dpitch, src, spitch, run_bytes, rows,
                       stream):
        import ctypes
        for r in range(rows):
            ctypes.memmove(self._real(dst + r * dpitch),
                           self._real(src + r * spitch), run_bytes)

    def synchronize(self, stream=0):
        return None


def _desc(name, nbytes, *, src=1, dst=1, src_ptr=None, dst_ptr=None):
    return wx.XchgDesc(
        tag="weights_0", src_rank=src, dst_rank=dst, param_name=name,
        kind=tp.FLAT, nbytes=nbytes, rows=1, run_bytes=nbytes, spitch=0,
        dpitch=0, src_ptr=src_ptr, dst_ptr=dst_ptr,
    )


class _TransportHarness(unittest.TestCase):
    """One lane, real shared buffer, real byte moves, no CUDA."""

    def setUp(self):
        # 2026-09-15: the per-unit sha256 is a DEVELOPMENT witness now
        # (SEQ_UNIT_DIGEST_ENV, default off on the metal); the desk mutants
        # below are its reason to exist, so arm it here.
        self._digest_env_before = os.environ.get(bx.SEQ_UNIT_DIGEST_ENV)
        os.environ[bx.SEQ_UNIT_DIGEST_ENV] = "1"
        self.root = tempfile.mkdtemp(prefix="weg2-seq-")
        self.nonce = f"seq1378{os.getpid()}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)
        self.ops = _MemOps()
        self.lines = []

    def tearDown(self):
        import shutil
        if self._digest_env_before is None:
            os.environ.pop(bx.SEQ_UNIT_DIGEST_ENV, None)
        else:
            os.environ[bx.SEQ_UNIT_DIGEST_ENV] = self._digest_env_before
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except BaseException:
            pass

    def _vram_pair(self, nbytes, i):
        """A deposit-side and a collect-side stand-in, both really backed."""
        src, dst = 0x1000 + i * 512, 0x8000 + i * 512
        self.ops.hook(src, nbytes)
        self.ops.hook(dst, nbytes)
        return src, dst

    def _deposit(self, descs):
        return bx.run_sequential_units(
            descs, self.ops, self.nonce, slot_bytes=1 << 30,
            shm_root=self.root, phase=bx.PHASE_DEPOSIT,
            log=self.lines.append, card=1)

    def _collect(self, descs, dst_digest_fn=None):
        return bx.run_sequential_units(
            descs, self.ops, self.nonce, slot_bytes=1 << 30,
            shm_root=self.root, phase=bx.PHASE_COLLECT,
            dst_digest_fn=dst_digest_fn, log=self.lines.append, card=1)


class Mutant1TransportRotatedUnitsDie(_TransportHarness):
    def test_rotated_units_die_on_the_buffer_digest(self):
        """MUTANT 1 (the TRANSPORT witness): the collect's unit list is
        ROTATED by one -- its piece 0 carries piece1's name/size and reads
        the wrong window -- dies on the sha256 window digest vs the
        deposit's record."""
        nbytes = [64, 128]
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xA0 + i]) * n)
            descs.append(_desc(f"unit{i}", n, src_ptr=src, dst_ptr=dst))
        rc = self._deposit(descs)
        self.assertEqual(rc, "", f"the deposit: {rc}")
        rotated = [_desc(descs[1].param_name, descs[1].nbytes,
                         src_ptr=descs[1].src_ptr, dst_ptr=descs[1].dst_ptr),
                   _desc(descs[0].param_name, descs[0].nbytes,
                         src_ptr=descs[0].src_ptr, dst_ptr=descs[0].dst_ptr)]
        rc = self._collect(rotated)
        # weg2xsn84: a rotated list now dies one check EARLIER, on the
        # record's identity (unit 0's record names unit0, the collect asks
        # for unit1) -- the digest witness behind it is unchanged and is
        # what Mutant3 leaves standing for a same-name, same-size unit.
        self.assertTrue("digest mismatch" in rc or "unit identity mismatch" in rc,
                        f"the rotated units must die before the copy-out: "
                        f"{rc!r}")

    def test_an_honest_list_passes_and_moves_the_bytes(self):
        """THE COUNTERPROBE the old form could not run: the unrotated list
        moves the deposit's own bytes to their own destinations.

        Red on 69f727dac2: the buffer was private, so this passed with ZERO
        bytes crossing (both sides digested their own zeros)."""
        nbytes = [64, 128, 32]
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xC0 + i]) * n)
            descs.append(_desc(f"unit{i}", n, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        self.assertEqual(self._collect(descs), "")
        for i, d in enumerate(descs):
            self.assertEqual(
                self.ops.read(int(d.dst_ptr), int(d.nbytes)),
                bytes([0xC0 + i]) * int(d.nbytes),
                f"piece {i}: the placement must be exact")


class Mutant2PlacementSwappedDestinationsDie(_TransportHarness):
    def test_swapped_destinations_die_on_the_placement_witness(self):
        """MUTANT 2 (the PLACEMENT witness): the DESTINATION ADDRESSES are
        swapped with IDENTICAL buffer reads -- the window digest passes
        green-falsely and the bytes land at each other's place.  Dies on the
        destination digest vs the deposit's record."""
        nbytes = [64, 64]  # equal: the swap must not be a buffer overflow,
        # it must be a PLACEMENT error the witness sees
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xB0 + i]) * n)
            descs.append(_desc(f"unit{i}", n, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        swapped = [_desc("unit0", 64, src_ptr=descs[0].src_ptr,
                         dst_ptr=descs[1].dst_ptr),
                   _desc("unit1", 64, src_ptr=descs[1].src_ptr,
                         dst_ptr=descs[0].dst_ptr)]

        def witness(addr, count):
            # THE PLACEMENT WITNESS: what this destination holds must be the
            # bytes of the tensor this destination is FOR -- read from the
            # SOURCE that owns that tensor, not from the deposit's record
            # (which follows the piece and would agree with whatever landed
            # there).  This is the desk form of the SEAM-DIGEST's
            # position-weighted fold, which is why the swap cannot hide.
            return self.ops.digest(self.src_of[addr], count)

        self.src_of = {int(d.dst_ptr): int(d.src_ptr) for d in descs}

        rc = self._collect(swapped, dst_digest_fn=witness)
        self.assertIn("placement mismatch", rc,
                      f"the swapped destinations must die on the placement "
                      f"witness: {rc!r}")


class LaneBufferIsTheLanesOwnSum(_TransportHarness):
    """#1378 xsn56 -- THE LANE'S BUFFER IS THE LANE'S OWN SUM.

    Operator order 2026-09-15: *"er limitiert die groesse des ringpuffers wohl
    immernoch."*  Boot weg2xsn55 refused with ``the lane's bytes do not fit one
    buffer (slot_bytes=756323776 produced 3 batches)``: the call site handed
    over ``terms.buffer_bytes``, which prices the WIDEST SINGLE LAYER, while a
    PP-form lane carries a whole band (114 descs, ~2.16 GiB measured).

    **WHY THE EXISTING TESTS COULD NOT CATCH IT:** every one of them passes
    ``slot_bytes=1 << 30`` -- a generous constant that is neither the priced
    number nor the lane's sum, so the harness never ran the production value.
    A fixture that is roomier than production is green by vacancy; this test
    drives BOTH real numbers instead.
    """

    #: Sizes stay inside the harness's 512-byte address stride
    #: (``_vram_pair`` hands out ``0x1000 + i * 512``): a 4096-byte piece made
    #: piece 1's window overlap piece 0's and aborted the process inside
    #: ``hook`` -- my own fixture defect, found by running it.  The property
    #: under test needs only ``sum > widest``, not large numbers.
    _SIZES = (128, 128, 128)

    def _band(self):
        descs = []
        for i, n in enumerate(self._SIZES):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xB0 + i]) * n)
            descs.append(_desc(f"band{i}", n, src_ptr=src, dst_ptr=dst))
        return descs

    def test_the_old_pricing_dies_and_the_lane_sum_carries(self):
        widest, lane_sum = max(self._SIZES), sum(self._SIZES)
        self.assertGreater(lane_sum, widest, "fixture must model a BAND")
        # ONE set of descriptors for both calls: `_band` hooks its windows, and
        # hooking the same address twice aborts the process.
        descs = self._band()

        # THE DYING MUTANT -- the shipped call site's number until xsn56.  The
        # refusal is raised before any semaphore is posted, so the second call
        # below starts from an untouched handshake.
        with self.assertRaises(ValueError) as caught:
            bx.run_sequential_units(
                descs, self.ops, self.nonce, slot_bytes=widest,
                shm_root=self.root, phase=bx.PHASE_DEPOSIT,
                log=self.lines.append, card=1)
        self.assertIn("do not fit one buffer", str(caught.exception))

        # THE PRODUCTION FORM -- the lane's own sum.  `batch_descs` packs
        # byte-exactly, so the sum is the exact lower bound for one batch:
        # one fewer byte must already fail, which the mutant above shows.
        self.lines.clear()
        out = bx.run_sequential_units(
            descs, self.ops, self.nonce, slot_bytes=lane_sum,
            shm_root=self.root, phase=bx.PHASE_DEPOSIT,
            log=self.lines.append, card=1)
        self.assertNotIn("no units", str(out))
        self.assertTrue(
            any(f"slot_bytes={lane_sum}" in ln for ln in self.lines),
            f"the lane line must carry its buffer size: {self.lines!r}")


if __name__ == "__main__":
    unittest.main()


class _PinOps(_MemOps):
    """The address-honest ops WITH a pin: records every register/unregister so
    the pairing can be asserted. `fail_register` makes the pin raise the way
    cudart does on an already-registered range (rc=712)."""

    def __init__(self, fail_register=False):
        super().__init__()
        self.pins = []
        self.fail_register = fail_register

    def host_register(self, ptr, nbytes, flags):
        if self.fail_register:
            raise RuntimeError("cudaHostRegister rc=712 part or all of the "
                               "requested memory range is already mapped")
        self.pins.append(("register", int(ptr), int(nbytes)))

    def host_unregister(self, ptr):
        self.pins.append(("unregister", int(ptr)))


class TheLanePinIsUnregisteredBeforeTheBufferGoes(_TransportHarness):
    """#1378 xsn70: P rank 0 pinned lane c0's buffer, closed it WITHOUT
    cudaHostUnregister, and the next two lanes' mmaps landed on the same VA
    range -- their register failed (already registered), 72 of p2's pieces
    read the stale pin, and p4's last piece, 1804 bytes past the stale range,
    died with cudaMemcpyAsync rc=1. Mutant: the shipped close, no unregister."""

    def setUp(self):
        # 2026-09-15: this witness is about the TRANSIENT form (register and
        # unregister per call); the shipped default keeps a lane's buffer
        # registered for the process (SEQ_PERSIST_BUFFERS_ENV), see below.
        os.environ[bx.SEQ_PERSIST_BUFFERS_ENV] = "0"
        super().setUp()
        self.ops = _PinOps()

    def test_every_register_has_its_unregister_at_the_same_address(self):
        nbytes = [64, 128, 32]
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xA0 + i]) * n)
            descs.append(_desc(f"unit{i}", n, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        self.assertEqual(self._collect(descs), "")
        regs = [p for p in self.ops.pins if p[0] == "register"]
        unregs = [p for p in self.ops.pins if p[0] == "unregister"]
        self.assertEqual(len(regs), 2, self.ops.pins)          # one lane, two phases
        self.assertEqual(len(unregs), 2, self.ops.pins)
        for r, u in zip(regs, unregs):
            self.assertEqual(r[1], u[1], "unregister must name the registered address")
        # order per phase: register ... unregister
        kinds = [p[0] for p in self.ops.pins]
        self.assertEqual(kinds, ["register", "unregister"] * 2, kinds)

    def test_a_failed_pin_refuses_the_lane_instead_of_copying_unpinned(self):
        self.ops = _PinOps(fail_register=True)
        src, dst = self._vram_pair(64, 0)
        self.ops.write(src, b"\x11" * 64)
        rc = self._deposit([_desc("unit0", 64, src_ptr=src, dst_ptr=dst)])
        self.assertIn("host_register failed", rc)
        self.assertTrue(any("registered=no(RuntimeError)" in ln for ln in self.lines))


class Mutant3UnitIdentityMismatchDies(_TransportHarness):
    """weg2xsn84: the source paused in the interleaved order (weights_4
    first), the destination resumed in the natural one (weights_0 first),
    and D collected PP0's weights_4 units as weights_0's with every window
    digest MATCHING -- same bytes, wrong tensor. The deposit record names
    the tensor and its tag; a collect whose desc names another one must
    refuse BEFORE the copy-out. MUTANT: the shipped collect, which checked
    the digest only."""

    def test_same_bytes_other_name_die_on_the_record_identity(self):
        nbytes = [64, 128]
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xD0 + i]) * n)
            descs.append(_desc(f"layer4.unit{i}", n, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        other = [_desc(f"layer0.unit{i}", d.nbytes, src_ptr=d.src_ptr,
                       dst_ptr=d.dst_ptr) for i, d in enumerate(descs)]
        before = [self.ops.read(int(d.dst_ptr), int(d.nbytes)) for d in other]
        rc = self._collect(other)
        self.assertIn("unit identity mismatch", rc, rc)
        self.assertIn("layer4.unit0", rc)
        self.assertIn("layer0.unit0", rc)
        for d, b in zip(other, before):
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)), b,
                             "refused BEFORE the copy-out: no byte moved")

    def test_same_name_other_tag_dies_too(self):
        n = 96
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x5a" * n)
        dep = _desc("shared.name", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([dep]), "")
        col = wx.XchgDesc(
            tag="weights_4", src_rank=1, dst_rank=1, param_name="shared.name",
            kind=tp.FLAT, nbytes=n, rows=1, run_bytes=n, spitch=0, dpitch=0,
            src_ptr=src, dst_ptr=dst)
        rc = self._collect([col])
        self.assertIn("unit identity mismatch", rc, rc)
        self.assertIn("tag=weights_0", rc)
        self.assertIn("tag=weights_4", rc)

    def test_the_matching_identity_still_passes(self):
        n = 48
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x77" * n)
        d = _desc("same.tensor", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")
        self.assertEqual(self._collect([d]), "")
        self.assertEqual(self.ops.read(dst, n), b"\x77" * n)


class NoWriteConsumesButDoesNotWrite(_TransportHarness):
    """weg2xsn86: a MEASURED target share (the draft's embed on D IS the
    target's tensor, region `weights`, still paused while `weights_draft`
    is collected) must be CONSUMED -- handshake, identity and transport
    digest as for every unit, so the two sides' unit indices stay aligned
    -- and NOT written. MUTANT: the shipped collect, which wrote every unit
    it consumed (SIGSEGV in cuMemcpyAsync at draft unit 2, TP0/1/2)."""

    def test_the_no_write_unit_is_consumed_and_its_destination_untouched(self):
        nbytes = [64, 128, 32]
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xE0 + i]) * n)
            descs.append(_desc(f"unit{i}", n, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        before = self.ops.read(int(descs[1].dst_ptr), nbytes[1])
        rc = bx.run_sequential_units(
            descs, self.ops, self.nonce, slot_bytes=1 << 30,
            shm_root=self.root, phase=bx.PHASE_COLLECT,
            no_write={("weights_0", "unit1")}, log=self.lines.append, card=1)
        self.assertEqual(rc, "", rc)
        self.assertEqual(self.ops.read(int(descs[0].dst_ptr), nbytes[0]),
                         bytes([0xE0]) * nbytes[0])
        self.assertEqual(self.ops.read(int(descs[2].dst_ptr), nbytes[2]),
                         bytes([0xE2]) * nbytes[2])
        self.assertEqual(self.ops.read(int(descs[1].dst_ptr), nbytes[1]),
                         before, "the no-write unit's destination is untouched")
        self.assertTrue(any("NO-WRITE" in ln and "unit1" in ln
                            for ln in self.lines))

    def test_no_write_is_keyed_by_tag_too(self):
        """The target's OWN `weights` leg carries the same name -- a bare
        name key would skip that write as well."""
        n = 40
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x11" * n)
        d = _desc("model.embed_tokens.weight", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")
        rc = bx.run_sequential_units(
            [d], self.ops, self.nonce, slot_bytes=1 << 30,
            shm_root=self.root, phase=bx.PHASE_COLLECT,
            no_write={("weights_draft", "model.embed_tokens.weight")},
            log=self.lines.append, card=1)
        self.assertEqual(rc, "", rc)
        self.assertEqual(self.ops.read(dst, n), b"\x11" * n,
                         "tag weights_0 is not in the no-write set: written")


class _ProbingMemOps(_MemOps):
    """A driver-style probe on top of the host fake: every address is device
    memory except the listed fake addresses (rc 0, type 0, device -1: a
    reserved, unmapped VMM range -- a PAUSED region)."""

    def __init__(self, unknown=()):
        super().__init__()
        self.unknown = set(int(a) for a in unknown)
        self.copies = []

    def ptr_attrs(self, addr):
        for fake in self.unknown:
            if fake <= int(addr) < fake + self.vram_size.get(fake, 0):
                return (0, 0, -1)
        return (0, 2, 0)

    def memcpy_async(self, dst, src, nbytes, stream):
        self.copies.append(int(dst))
        super().memcpy_async(dst, src, nbytes, stream)


class UnmappedDestinationIsRefusedByName(_TransportHarness):
    """fnFL2x34 (23.09.): unit 2 of the draft band, lm_head.weight_packed,
    was written into the PAUSED target head (region `weights`, type 0 to the
    driver) -- SIGSEGV in cudaMemcpyAsync two lines after the i==0 probe
    said type=2 for unit 0. MUTANT: the shipped collect, which probed unit 0
    only and logged instead of refusing."""

    def setUp(self):
        super().setUp()
        self.ops = _ProbingMemOps()

    def test_the_unmapped_unit_is_named_and_nothing_after_it_is_written(self):
        nbytes = [64, 128, 32]
        descs = []
        for i, n in enumerate(nbytes):
            src, dst = self._vram_pair(n, i)
            self.ops.write(src, bytes([0xA0 + i]) * n)
            descs.append(_desc(f"unit{i}", n, src_ptr=src, dst_ptr=dst))
        descs[2] = _desc("lm_head.weight_packed", nbytes[2],
                         src_ptr=descs[2].src_ptr, dst_ptr=descs[2].dst_ptr)
        self.ops.unknown = {int(descs[2].dst_ptr)}
        self.assertEqual(self._deposit(descs), "")
        before = self.ops.read(int(descs[2].dst_ptr), nbytes[2])
        rc = bx.run_sequential_units(
            descs, self.ops, self.nonce, slot_bytes=1 << 30,
            shm_root=self.root, phase=bx.PHASE_COLLECT,
            log=self.lines.append, card=1)
        self.assertIn("lm_head.weight_packed", rc)
        self.assertIn("not mapped device memory", rc)
        self.assertIn("type=0", rc)
        self.assertNotIn(int(descs[2].dst_ptr), self.ops.copies,
                         "the unknown destination was never copied into")
        self.assertEqual(self.ops.read(int(descs[2].dst_ptr), nbytes[2]), before)

    def test_a_probe_less_ops_keeps_the_shipped_form(self):
        self.assertIsNone(tp.dst_pointer_probe(_MemOps()))


class DigestOffByDefaultStillMovesAndChecksIdentity(_TransportHarness):
    """The default form (flag unset): no sha256 on either side, the record
    carries digest="" and the collect prints digest=off; the identity guard
    and the bytes are untouched. MUTANT: a collect that still hashed."""

    def test_default_off_moves_bytes_and_keeps_the_identity_guard(self):
        os.environ.pop(bx.SEQ_UNIT_DIGEST_ENV, None)
        self.assertFalse(bx.seq_unit_digest_armed())
        n = 96
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x42" * n)
        d = _desc("plain.unit", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")
        self.assertEqual(self._collect([d]), "")
        self.assertEqual(self.ops.read(dst, n), b"\x42" * n)
        self.assertTrue(any("digest=off" in ln for ln in self.lines), self.lines[-3:])
        other = _desc("other.unit", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")
        self.assertIn("unit identity mismatch", self._collect([other]))

    def test_buffer_slot_names_a_second_file(self):
        self.assertEqual(bx.seq_lane_file_name("c0", 0), "c0")
        self.assertEqual(bx.seq_lane_file_name("c0", 1), "c0_s1")
        n = 32
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x07" * n)
        d = _desc("slot.unit", n, src_ptr=src, dst_ptr=dst)
        rc = bx.run_sequential_units(
            [d], self.ops, self.nonce, slot_bytes=1 << 30, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, buffer_slot=1, log=self.lines.append, card=1)
        self.assertEqual(rc, "")
        self.assertTrue(os.path.exists(
            bx.sequential_buffer_path(self.nonce, self.root, lane="c1_s1")))
        rc = bx.run_sequential_units(
            [d], self.ops, self.nonce, slot_bytes=1 << 30, shm_root=self.root,
            phase=bx.PHASE_COLLECT, buffer_slot=1, log=self.lines.append, card=1)
        self.assertEqual(rc, "")
        self.assertEqual(self.ops.read(dst, n), b"\x07" * n)


class PrimeDrainTakesEveryLeftoverCredit(_TransportHarness):
    """weg2xsn88: under buffer depth 2 a role switch leaves TWO drain
    credits; a prime that took one let the new depositor run a tag too far
    ahead (W68 unit identity mismatch on p3, W90 moved=66 on PP0).
    MUTANT: the shipped single trywait."""

    def test_prime_drains_all_credits(self):
        rv = bx.CrossSlotRendezvous(tp.SemSet(self.nonce), None, card=1)
        rv.post_drained(tag="a")
        rv.post_drained(tag="b")  # armed 1 + 2 posts = 3 credits
        self.assertTrue(rv.prime_drain())
        self.assertEqual(rv.last_primed, 3)
        self.assertFalse(rv.prime_drain())
        self.assertEqual(rv.last_primed, 0)
        self.assertFalse(rv._wait(rv._DRAIN_SLOT, "empty", 0.0))


class _IpcOps(_MemOps):
    """A device with cudaMalloc + CUDA IPC, faked: the staging is a host
    block, the 64-byte handle carries its fake address."""

    def __init__(self):
        super().__init__()
        self.freed = []
        self._next = 0x100000

    def raw_malloc(self, device, nbytes):
        addr = self._next
        self._next += (nbytes + 4095) & ~4095
        self.hook(addr, nbytes)
        return addr

    def raw_free(self, ptr):
        self.freed.append(int(ptr))

    def ipc_get_handle(self, ptr):
        return int(ptr).to_bytes(8, "little") + b"\0" * 56

    def ipc_open_handle(self, handle):
        return int.from_bytes(handle[:8], "little")

    def ipc_close_handle(self, ptr):
        self.closed = int(ptr)


class PersistentBuffersAndOnCardIpc(_TransportHarness):
    """2026-09-15 (Nutzer-Order: persistente Puffer, On-Card per IPC)."""

    def setUp(self):
        super().setUp()
        os.environ.pop(bx.SEQ_PERSIST_BUFFERS_ENV, None)  # default: on
        os.environ.pop(bx.SEQ_ONCARD_IPC_ENV, None)       # default: on
        # weg2xsn268/269: host lanes run pageable by default (cudaHostUnregister
        # was the 12.5-s stall); this class tests the REGISTERED A/B form.
        os.environ[bx.SEQ_HOST_REGISTER_ENV] = "1"

    def tearDown(self):
        os.environ.pop(bx.SEQ_HOST_REGISTER_ENV, None)
        super().tearDown()

    def test_the_lane_buffer_is_registered_once_and_reused(self):
        ops = _PinOps()
        self.ops = ops
        n = 64
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x21" * n)
        d = _desc("persist.unit", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")
        self.assertEqual(self._collect([d]), "")
        self.assertEqual(self.ops.read(dst, n), b"\x21" * n)
        self.ops.write(src, b"\x22" * n)
        self.assertEqual(self._deposit([d]), "")
        self.assertEqual(self._collect([d]), "")
        self.assertEqual(self.ops.read(dst, n), b"\x22" * n)
        regs = [c for c in ops.pins if c[0] == "register"]
        unregs = [c for c in ops.pins if c[0] == "unregister"]
        self.assertEqual(len(regs), 1, ops.pins)
        self.assertEqual(unregs, [], "persistent: never unregistered per tag")
        self.assertTrue(any("persist lane=c1 reuse" in ln for ln in self.lines))

    def test_on_card_lane_stages_by_ipc_and_the_host_path_stays_for_cross(self):
        ops = _IpcOps()
        self.ops = ops
        n = 96
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x33" * n)
        d = _desc("ipc.unit", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")   # card=1: on-card lane
        self.assertTrue(any("ipc lane=c1 phase=deposit stage=" in ln
                            for ln in self.lines), self.lines[-4:])
        self.assertEqual(self._collect([d]), "")
        self.assertTrue(any("ipc lane=c1 phase=collect opened=" in ln
                            for ln in self.lines))
        self.assertEqual(self.ops.read(dst, n), b"\x33" * n)
        self.assertEqual(getattr(ops, "closed", None) is not None, True)
        self.assertEqual(bx.release_stage_buffers(self.nonce, log=self.lines.append), 1)
        self.assertEqual(len(ops.freed), 1)

    def test_ipc_unavailable_falls_back_to_the_host_path(self):
        class _Broken(_IpcOps):
            def raw_malloc(self, device, nbytes):
                raise RuntimeError("cudaMalloc: out of memory")
        self.ops = _Broken()
        n = 40
        src, dst = self._vram_pair(n, 0)
        self.ops.write(src, b"\x44" * n)
        d = _desc("fallback.unit", n, src_ptr=src, dst_ptr=dst)
        self.assertEqual(self._deposit([d]), "")
        self.assertTrue(any("phase=deposit UNAVAILABLE" in ln for ln in self.lines))
        self.assertEqual(self._collect([d]), "")
        self.assertEqual(self.ops.read(dst, n), b"\x44" * n)


class _DeferredOps(_MemOps):
    """Order point 2 (batched syncs): copies are ASYNC for real here -- a
    `memcpy_async` only queues, the bytes move at `synchronize`. A deposit
    that digests/records/posts before its sync hashes the unwritten window;
    a collect that runs the placement witness before its sync reads the old
    destination. Both mutants die on this ops, the per-unit form and the
    batched form both pass."""

    def __init__(self):
        super().__init__()
        self.pending = []
        self.syncs = 0
        self.copies = 0

    def memcpy_async(self, dst, src, nbytes, stream):
        self.copies += 1
        self.pending.append(("f", dst, src, nbytes))

    def memcpy2d_async(self, dst, dpitch, src, spitch, run_bytes, rows, stream):
        self.copies += 1
        self.pending.append(("2d", dst, dpitch, src, spitch, run_bytes, rows))

    def synchronize(self, stream=0):
        self.syncs += 1
        pend, self.pending = self.pending, []
        for op in pend:
            if op[0] == "f":
                _MemOps.memcpy_async(self, op[1], op[2], op[3], stream)
            else:
                _MemOps.memcpy2d_async(self, *op[1:], stream)


class BatchedSyncsMoveTheBytesAndSyncOncePerBatch(_TransportHarness):
    """Order point 2: N units, one sync per batch, every witness intact."""

    def _env(self, mib, units):
        os.environ[bx.SEQ_SYNC_BATCH_MIB_ENV] = str(mib)
        os.environ[bx.SEQ_SYNC_BATCH_UNITS_ENV] = str(units)
        self.addCleanup(os.environ.pop, bx.SEQ_SYNC_BATCH_MIB_ENV, None)
        self.addCleanup(os.environ.pop, bx.SEQ_SYNC_BATCH_UNITS_ENV, None)

    def _run(self, n_units=10, nbytes=256):
        self.ops = _DeferredOps()
        descs, payloads = [], []
        for i in range(n_units):
            src, dst = self._vram_pair(nbytes, i)
            data = bytes([(i * 7 + 3) % 251]) * nbytes
            self.ops.write(src, data)
            payloads.append((dst, data))
            descs.append(_desc(f"u{i}", nbytes, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        dep_syncs = self.ops.syncs
        self.assertEqual(self._collect(descs, dst_digest_fn=self.ops.digest), "")
        col_syncs = self.ops.syncs - dep_syncs
        for dst, data in payloads:
            self.assertEqual(self.ops.read(dst, nbytes), data)
        return dep_syncs, col_syncs

    def test_units_batch_bound_gives_ceil_n_over_k_syncs(self):
        self._env(1024, 4)
        dep, col = self._run(n_units=10)
        self.assertEqual((dep, col), (3, 3))
        lt = [l for l in self.lines if "lane-time" in l]
        self.assertTrue(any("syncs=3 batch=4u/1024MiB" in l for l in lt), lt)

    def test_bytes_batch_bound_closes_the_batch(self):
        # 4096-byte units, batch bound 1 MiB -> 256 units per batch; 10 units
        # fit one batch; with 1 MiB and 300 units the bound is 256.
        self._env(1, 4096)
        dep, col = self._run(n_units=10)
        self.assertEqual((dep, col), (1, 1))

    def test_units_one_is_the_per_unit_form(self):
        self._env(1024, 1)
        dep, col = self._run(n_units=6)
        self.assertEqual((dep, col), (6, 6))

    def test_mutant_post_before_sync_dies_on_the_transport_digest(self):
        """MUTANT: the deposit digests/posts before the batch sync. With
        async copies the deposit hashes an unwritten window and the collect
        (after the deposit's later sync) hashes the real bytes -> refusal."""
        self._env(1024, 4)
        real_sync = _DeferredOps.synchronize

        class _LateSyncOps(_DeferredOps):
            def synchronize(self, stream=0):
                # the mutant: the deposit's sync does nothing until the
                # collect side's first sync (models "post before sync")
                if not getattr(self, "_late", False):
                    self.syncs += 1
                    return None
                return real_sync(self, stream)

        self.ops = _LateSyncOps()
        descs = []
        for i in range(6):
            src, dst = self._vram_pair(256, i)
            self.ops.write(src, bytes([i + 1]) * 256)
            descs.append(_desc(f"m{i}", 256, src_ptr=src, dst_ptr=dst))
        self.assertEqual(self._deposit(descs), "")
        self.ops._late = True
        self.ops.pending, self.ops.pending_dep = [], self.ops.pending
        # flush the deposit's copies now (as if its sync had been late)
        for op in self.ops.pending_dep:
            _MemOps.memcpy_async(self.ops, op[1], op[2], op[3], 0)
        verdict = self._collect(descs, dst_digest_fn=self.ops.digest)
        self.assertIn("digest mismatch", verdict)


class SyncGroupsAreDeterministic(unittest.TestCase):
    def test_groups_close_on_units_or_bytes(self):
        from types import SimpleNamespace as NS
        pieces = [NS(nbytes=n) for n in (10, 10, 10, 50, 10, 10)]
        self.assertEqual(bx._sync_groups(pieces, 25, 100), [[0, 1], [2], [3], [4, 5]])
        self.assertEqual(bx._sync_groups(pieces, 10**9, 2), [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(bx._sync_groups(pieces, 10**9, 1), [[i] for i in range(6)])
        self.assertEqual(bx._sync_groups([], 1, 1), [])
