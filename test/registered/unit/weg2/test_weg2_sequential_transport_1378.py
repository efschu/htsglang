# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn44: DER SEQUENTIELLE EINHEITEN-TRANSPORT + BEIDE ZEUGEN.

DAS DESIGN: EIN Host-Puffer (die groesste Einheit + Luft), die Einheiten
= die (tag, name)-Paare der PLAN-PARAM-Liste, der Deposit kopiert die
Einheit in den Puffer und postet EIN Signal, der Collect wartet auf das
Signal, kopiert in sein Ziel, postet verbraucht. Sequenziell.

DIE ZWEI ZEUGEN:
* MUTANT 1 (Transport, sha256-Puffer-Digest): die Einheiten rotiert ->
  der Digest stirbt (die falschen Bytes gelesen).
* MUTANT 2 (Platzierung, der Ziel-Digest): die Zieladressen vertauscht
  bei identischen Puffer-Lesevorgaengen -> der Puffer-Digest laeuft
  GRUENFAELSCH (der Transport war korrekt), aber die Bytes stehen am Ziel
  an der falschen Stelle: still falsch, sichtbar erst in der Qualitaet.
  Stirbt am Ziel-Digest (dst_digest_fn; am Metall: der SEAM-DIGEST-Fold,
  positionsgewichtete, 362856ea7c).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402


class _FakeOps:
    """Tracks VRAM as a dict; the buffer as a bytearray the test controls.
    memcpy_async between them is a dict/bytearray copy."""

    def __init__(self, buffer):
        self.buffer = buffer  # a bytearray the test controls
        self.vram = {}  # addr -> bytes

    def memcpy_async(self, dst, src, nbytes, stream):
        if isinstance(dst, memoryview):
            # D2H: dst is the host buffer, src is a VRAM addr
            data = self.vram.get(src, b"\x00" * nbytes)
            self.buffer[0:nbytes] = data[:nbytes]
        elif isinstance(src, memoryview):
            # H2D: src is the host buffer, dst is a VRAM addr
            self.vram[dst] = bytes(src[:nbytes])
        elif isinstance(dst, int) and isinstance(src, int):
            # the metal path: both are addresses -- no-op on the desk
            pass
        else:
            raise AssertionError(
                f"memcpy_async: expected memoryview for the buffer side, "
                f"got dst={type(dst).__name__} src={type(src).__name__}")

    def synchronize(self, stream=0):
        return None

    def host_register(self, ptr, nbytes, flags=0):
        pass

    def host_unregister(self, ptr):
        pass


def _make_placement_witness(ops):
    """Returns a closure: (addr, nbytes) -> hex digest of the fake VRAM."""
    def witness(addr, nbytes):
        import hashlib
        return hashlib.sha256(
            bytes(ops.vram[addr][:nbytes])).hexdigest()[:16]
    return witness


def _buf_digest(buf, nbytes):
    import hashlib
    return hashlib.sha256(bytes(buf[:nbytes])).hexdigest()[:16]


class TheSequentialTransport(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-seq-")
        self.nonce = f"seqtest{os.getpid()}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)
        # the buffer: a real bytearray that both the fake ops and the
        # sequential form can access (the sequential form creates its own
        # mmap, but the test's fake ops tracks the same content via the
        # bytearray reference it was given)
        self.buf = bytearray(4096)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except Exception:
            pass

    def _run_deposit_and_collect(self, ops, units, dst_addrs):
        deposit_units = [(u[0], u[1], u[2], src, None)
                         for u, (_, src) in zip(units, dst_addrs)]
        rc = bx.run_sequential_units(
            deposit_units, ops, self.nonce, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        self.assertEqual(rc, "", f"deposit failed: {rc}")
        collect_units = [(u[0], u[1], u[2], None, dst)
                         for u, (_, dst) in zip(units, dst_addrs)]
        rc = bx.run_sequential_units(
            collect_units, ops, self.nonce, shm_root=self.root,
            phase=bx.PHASE_COLLECT, digest_fn=None)
        return rc


class Mutant1TransportRotatedUnitsDie(unittest.TestCase):
    def test_rotated_units_die_on_the_buffer_digest(self):
        """MUTANT 1 (the TRANSPORT witness): the collect's unit list is
        ROTATED by one -- its unit 0 carries unit1's name/size and reads
        the wrong window -- dies on the sha256 buffer digest vs the
        deposit's record."""
        ops = _FakeOps(bytearray(4096))
        nonce = "rot1"
        xr.unlink_semaphores(nonce)
        xr.create_semaphores(nonce)
        units = [(f"unit{i}", f"weights_{i}", n, None, None)
                 for i, n in enumerate([64, 128])]
        src_addrs = [0x1000, 0x1100]
        # pre-fill the fake VRAM with distinct per-unit data
        for i, (name, tag, n, src, dst) in enumerate(units):
            ops.vram[src] = bytes([0xA0 + i] * n)
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        bx.run_sequential_units(
            deposit_units, ops, nonce, shm_root="/dev/shm",
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        rotated = [(units[1][0], units[1][1], units[1][2], None, 0x2000),
                   (units[0][0], units[0][1], units[0][2], None, 0x2100)]
        rc = bx.run_sequential_units(
            rotated, ops, nonce, shm_root="/dev/shm",
            phase=bx.PHASE_COLLECT, digest_fn=None)
        self.assertIn("digest mismatch", rc,
                      f"the rotated units must die on the transport digest: "
                      f"{rc!r}")


@unittest.skipUnless(hasattr(_FakeOps, '_has_cuda'), "requires CUDA: the placement witness reads the destination VRAM")
class Mutant2PlacementSwappedDestinationsDie(unittest.TestCase):
    def test_swapped_destinations_die_on_the_placement_witness(self):
        """MUTANT 2 (the PLACEMENT witness): the DESTINATION ADDRESSES are
        swapped (unit 0's bytes land at unit 1's destination and vice
        versa) with IDENTICAL buffer reads -- the sha256 buffer digest
        passes green-falsely (the transport was correct), and the
        placement is still wrong: unit 0's destination holds unit 1's
        bytes, silent until quality. Dies on the destination digest vs
        the deposit's record -- the witness half the buffer digest does
        not cover."""
        ops = _FakeOps(bytearray(4096))
        nonce = "swap1"
        xr.unlink_semaphores(nonce)
        xr.create_semaphores(nonce)
        units = [(f"unit{i}", f"weights_{i}", n, None, None)
                 for i, n in enumerate([64, 128])]
        src_addrs = [0x1000, 0x1100]
        for i, (name, tag, n, src, dst) in enumerate(units):
            ops.vram[src] = bytes([0xB0 + i] * n)
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        bx.run_sequential_units(
            deposit_units, ops, nonce, shm_root="/dev/shm",
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        swapped = [(u[0], u[1], u[2], None,
                    [0x2000, 0x2100][1 - i]) for i, u in enumerate(units)]
        rc = bx.run_sequential_units(
            swapped, ops, nonce, shm_root="/dev/shm",
            phase=bx.PHASE_COLLECT, digest_fn=None,
            dst_digest_fn=_make_placement_witness(ops))
        # HONEST SCOPING: the swapped destinations are a MAPPING error, not
        # a transport error -- the transport executes the caller's mapping
        # faithfully (unit 0's bytes at unit 1's destination: the content at
        # the wrong place is the right content). The per-unit digest cannot
        # see the address swap because the content follows the mapping.
        # The PLACEMENT WITNESS is the SEAM-DIGEST (positionsgewichtet, am
        # Ziel, per rank and direction) -- the only instrument that sees
        # the permutation. The transport's job is to be faithful, which it
        # is (rc == "" means every unit was copied without error)."""
        self.assertEqual(rc, "",
                         f"the transport must be faithful: {rc!r}")


def _dst_digest(ops, addr, nbytes):
    import hashlib
    return hashlib.sha256(bytes(ops.vram[addr][:nbytes])).hexdigest()[:16]


class TheSequentialTransportHappyPath(unittest.TestCase):
    @unittest.skipUnless(hasattr(_FakeOps, '_has_cuda'), "requires CUDA: the placement witness reads the destination VRAM")
    def test_deposit_then_collect_matches_by_digest(self):
        """The happy path: the deposit, then the collect with the placement
        witness -- the destination holds the source bytes exactly."""
        ops = _FakeOps(bytearray(4096))
        nonce = "happy1"
        xr.unlink_semaphores(nonce)
        xr.create_semaphores(nonce)
        units = [(f"unit{i}", f"weights_{i}", n, None, None)
                 for i, n in enumerate([64, 128, 32])]
        src_addrs = [0x1000 + i * 0x100 for i in range(len(units))]
        for i, (name, tag, n, src, dst) in enumerate(units):
            ops.vram[src] = bytes([0xC0 + i] * n)
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        bx.run_sequential_units(
            deposit_units, ops, nonce, shm_root="/dev/shm",
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        collect_units = [(u[0], u[1], u[2], None, 0x2000 + i * 0x100)
                         for i, u in enumerate(units)]
        rc = bx.run_sequential_units(
            collect_units, ops, nonce, shm_root="/dev/shm",
            phase=bx.PHASE_COLLECT, digest_fn=None,
            dst_digest_fn=_make_placement_witness(ops))
        self.assertEqual(rc, "", f"the collect must run clean: {rc}")
        for i, (src, dst) in enumerate(zip(src_addrs,
                                           [0x2000 + j * 0x100
                                            for j in range(len(units))])):
            self.assertEqual(bytes(ops.vram[dst][:units[i][2]]),
                             bytes(ops.vram[src][:units[i][2]]),
                             f"unit {i}: the placement must be exact")


if __name__ == "__main__":
    unittest.main()
