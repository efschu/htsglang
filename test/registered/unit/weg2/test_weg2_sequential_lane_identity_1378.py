# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn53: THE LANE'S OWN IDENTITY, the shared buffer, and W15's teeth.

DIE DREI BEWEISE DES SITZES, alle ohne CUDA (``CUDA_VISIBLE_DEVICES=""``):

* T1/T3 -- DIE IDENTITAET.  Der resolved Semaphore-Name wird JE SEITE eine
  Zeile geloggt (deposit und collect) und die beiden Zeilen muessen denselben
  Namen buchstabieren.  Der xsn52-Wand war ``post(pair=0, ...)`` -- eine
  fremde Cross-Semaphore; der xsn53-Desk-Befund war ``diagonal_post(i, ...)``
  mit dem EINHEITSINDEX als Karte, was an der vierten Einheit an der
  Grenze stirbt und fuer die ersten drei alle drei Karten belegt.  Beide
  Formen sind hier tot: der Name kommt aus dem Lane-Schluessel
  (``pair=`` fuer Cross, ``card=`` fuer Diagonal), nicht aus der Liste.
* T6 -- DER PUFFER IST GETEILT.  Zwei echte Prozesse, ein tmpfs-mmap: was
  der Deposit schreibt, liest der Collect.  69f727dac2 hatte den Puffer
  privatisiert (``create_string_buffer``), sodass der Collect seine eigenen
  Nullen digestierte -- der Desk-Test sah es nicht, weil beide Seiten dann
  ueber Nullen denselben Digest bekommen.
* T5 -- W15 HAT ZAEHNE.  ``cross_sem_name(src == dst)`` verweigert die
  Diagonale BY NAME.  Der sterbende Mutant ist der Guard selbst: zieht man
  die Verweigerung heraus, laeuft der Test rot (siehe
  ``test_w15_refusal_is_load_bearing``).

Die Namensaufloesung braucht keine Karte und kein VRAM: sie ist reine
String-Arithmetik ueber dem Lane-Schluessel.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402


def _desc(name, nbytes, *, src=0, dst=0, tag="weights_0", src_ptr=None,
          dst_ptr=None, kind=None, rows=1, run_bytes=None, spitch=0, dpitch=0):
    """One FLAT descriptor, shaped the way ``build_plan`` emits them."""
    return wx.XchgDesc(
        tag=tag, src_rank=src, dst_rank=dst, param_name=name,
        kind=kind or tp.FLAT, nbytes=nbytes, rows=rows,
        run_bytes=run_bytes if run_bytes is not None else nbytes,
        spitch=spitch, dpitch=dpitch, src_ptr=src_ptr, dst_ptr=dst_ptr,
    )


class _MemOps:
    """memcpy between a VRAM stand-in and the SHARED buffer, by address.

    The metal path hands the transport an ADDRESS for the buffer (the mmap's
    host virtual address) and an ADDRESS for the tensor.  This fake allocates
    its "VRAM" as real memory at a real address, so one ``ctypes.memmove`` per
    copy covers both sides -- exactly what cudaMemcpy does for a pageable host
    destination.  A byte that takes the wrong path lands in the wrong window
    and the digest sees it; a byte that never crosses the process boundary
    never arrives, which is what the two-process test asserts.
    """

    def __init__(self):
        self.vram = {}
        self.vram_real = {}
        self.vram_size = {}

    def hook(self, addr, nbytes):
        """Allocate the VRAM stand-in for the fake device address ``addr``."""
        import ctypes
        buf = ctypes.create_string_buffer(nbytes)
        self.vram[addr] = buf
        self.vram_real[addr] = ctypes.addressof(buf)
        self.vram_size[addr] = nbytes

    def memcpy_async(self, dst, src, nbytes, stream):
        import ctypes
        ctypes.memmove(self._real(dst), self._real(src), nbytes)

    def memcpy2d_async(self, dst, dpitch, src, spitch, run_bytes, rows,
                       stream):
        import ctypes
        for r in range(rows):
            ctypes.memmove(self._real(dst + r * dpitch),
                           self._real(src + r * spitch), run_bytes)

    def _real(self, addr):
        for fake, real in self.vram_real.items():
            if fake <= addr < fake + self.vram_size[fake]:
                return real + (addr - fake)
        return addr  # not a fake VRAM address: the shared mmap itself

    def digest(self, addr, nbytes):
        import hashlib
        return hashlib.sha256(self.read(addr, nbytes)).hexdigest()[:16]

    def read(self, addr, nbytes):
        import ctypes
        return ctypes.string_at(self._real(addr), nbytes)

    def write(self, addr, data):
        import ctypes
        ctypes.memmove(self._real(addr), bytes(data), len(data))

    def synchronize(self, stream=0):
        return None


class _LaneHarness(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-seq-lane-")
        self.nonce = f"xsn53{os.getpid()}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)
        self.lines = []

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except BaseException:
            pass

    def _log(self, msg):
        self.lines.append(msg)

    def _run(self, descs, phase, ops=None, **kw):
        return bx.run_sequential_units(
            descs, ops or _MemOps(), self.nonce, slot_bytes=1 << 30,
            shm_root=self.root, phase=phase, log=self._log, **kw)

    def _handshakes(self):
        return [ln.split("handshake=")[1].split()[0] for ln in self.lines
                if "handshake=" in ln]


class TheLaneNamesItsOwnHandshake(_LaneHarness):
    def test_the_diagonal_resolves_the_card_not_the_unit_index(self):
        """T1: a diagonal lane's name carries the CARD, not the unit index.

        Red on 2234bbac0e: that code passed the UNIT INDEX as the card, so a
        lane on card 1 resolved ``-card0-`` (unit 0) and died at unit 3 with
        "card=3 is not one of this rig's 3 cards".
        """
        ops = _MemOps()
        descs = []
        for i in range(6):
            nbytes = 64 + 16 * i
            ops.hook(0x1000 + i * 512, nbytes)   # the deposit's own VRAM
            ops.hook(0x8000 + i * 512, nbytes)   # the collect's own VRAM
            ops.write(0x1000 + i * 512, bytes([0xA0 + i]) * nbytes)
            descs.append(_desc(f"t{i}", nbytes, src=1, dst=1,
                               src_ptr=0x1000 + i * 512,
                               dst_ptr=0x8000 + i * 512))
        rc = bx.run_sequential_units(descs, ops, self.nonce,
                                     slot_bytes=1 << 30, shm_root=self.root,
                                     phase=bx.PHASE_DEPOSIT, log=self._log,
                                     card=1)
        self.assertEqual(rc, "", f"the deposit: {rc}")
        rc = bx.run_sequential_units(descs, ops, self.nonce,
                                     slot_bytes=1 << 30, shm_root=self.root,
                                     phase=bx.PHASE_COLLECT, log=self._log,
                                     card=1)
        self.assertEqual(rc, "", f"the collect: {rc}")
        for i, d in enumerate(descs):
            self.assertEqual(ops.read(int(d.dst_ptr), int(d.nbytes)),
                             bytes([0xA0 + i]) * int(d.nbytes),
                             f"piece {i}: the bytes must arrive at their own "
                             f"window's destination")
        names = self._handshakes()
        self.assertEqual(len(names), 2, f"one line per side: {self.lines}")
        self.assertEqual(names[0], names[1],
                         "deposit and collect must spell the same name")
        self.assertIn("-card1-0-full", names[0],
                      f"the CARD must name the handshake: {names[0]}")
        self.assertNotIn("-card0-", names[0])

    def test_the_cross_lane_resolves_the_cross_family(self):
        """T3: a cross lane's name is the directed pair's, not a diagonal."""
        ops = _MemOps()
        ops.hook(0x1000, 64)
        ops.hook(0x2000, 64)
        descs = [_desc("t", 64, src=0, dst=1, src_ptr=0x1000, dst_ptr=0x2000)]
        bx.run_sequential_units(descs, ops, self.nonce, slot_bytes=1 << 30,
                                shm_root=self.root, phase=bx.PHASE_DEPOSIT,
                                log=self._log, pair=0)
        self.assertIn("-0-1-0-full", self._handshakes()[0],
                      f"the cross name encodes both cards: {self.lines}")
        self.assertNotIn("-card", self._handshakes()[0])

    def test_more_pieces_than_cards_complete(self):
        """T2: the fourth piece must not bounds out on the card count.

        Red on 2234bbac0e: the unit index WAS the card, so piece 3 raised
        ValueError("card=3 is not one of this rig's 3 cards").
        """
        ops = _MemOps()
        descs = []
        for i in range(10):
            ops.hook(0x1000 + i * 128, 64)
            ops.hook(0x8000 + i * 128, 64)
            descs.append(_desc(f"t{i}", 64, src=2, dst=2,
                               src_ptr=0x1000 + i * 128,
                               dst_ptr=0x8000 + i * 128))
        rc = bx.run_sequential_units(descs, ops, self.nonce,
                                     slot_bytes=1 << 30, shm_root=self.root,
                                     phase=bx.PHASE_DEPOSIT, log=self._log,
                                     card=2)
        self.assertEqual(rc, "", f"ten pieces on one lane: {rc}")

    def test_an_ambiguous_lane_key_is_refused_by_name(self):
        """T4: the CrossSlotRendezvous contract, at this form's own door."""
        descs = [_desc("t", 64, src=1, dst=1, src_ptr=0x1000,
                       dst_ptr=0x2000)]
        with self.assertRaises(ValueError) as caught:
            self._run(descs, bx.PHASE_DEPOSIT, pair=0, card=1)
        self.assertIn("exactly one of pair=", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self._run(descs, bx.PHASE_DEPOSIT)
        self.assertIn("exactly one of pair=", str(caught.exception))


class TheBufferIsShared(_LaneHarness):
    def test_bytes_cross_processes_through_the_lane_buffer(self):
        """T6: two REAL processes, one tmpfs mmap.

        Red on 69f727dac2: the buffer was a private per-process allocation, so
        the collect read its own zero-filled pages and the two digests agreed
        on nothing but zeros.
        """
        ctx = mp.get_context("fork")
        pattern = bytes(range(256)) * 4
        result = ctx.Queue()

        def deposit(root, nonce, queue):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            ops = _MemOps()
            ops.hook(0x1000, len(pattern))
            ops.write(0x1000, pattern)
            descs = [_desc("t", len(pattern), src=1, dst=1, src_ptr=0x1000)]
            rc = bx.run_sequential_units(
                descs, ops, nonce, slot_bytes=1 << 30, shm_root=root,
                phase=bx.PHASE_DEPOSIT, log=lambda *a: None, card=1)
            queue.put(("deposit", rc))

        p = ctx.Process(target=deposit,
                        args=(self.root, self.nonce, result))
        p.start()
        p.join(60)
        self.assertEqual(p.exitcode, 0, "the deposit process died")
        kind, rc = result.get(timeout=10)
        self.assertEqual((kind, rc), ("deposit", ""), f"{kind}: {rc}")

        ops = _MemOps()
        ops.hook(0x2000, len(pattern))
        descs = [_desc("t", len(pattern), src=1, dst=1, dst_ptr=0x2000)]
        rc = self._run(descs, bx.PHASE_COLLECT, ops=ops, card=1)
        self.assertEqual(rc, "", f"the collect: {rc}")
        got = ops.read(0x2000, len(pattern))
        self.assertEqual(got, pattern,
                         "the collect must read the deposit's own bytes")


class W15HasTeeth(unittest.TestCase):
    def test_a_diagonal_id_on_the_cross_path_is_refused_by_name(self):
        """T5: src == dst is the diagonal and the cross path refuses it.

        Red on 2234bbac0e: ``sem_name`` checked only the RANGE of ``pair``, so
        a diagonal id in [0, 6) resolved a foreign cross name -- the xsn52
        hang, all six ranks on one semaphore.
        """
        with self.assertRaises(xr.Weg2XchgDiagonalHasNoCrossPair) as caught:
            xr.cross_sem_name("nonce", 0, 0, 0, "full")
        self.assertIn("W15", str(caught.exception))
        self.assertIn("diagonal_sem_name", str(caught.exception),
                      "the refusal must name the diagonal's own door")

    def test_a_card_the_rig_does_not_have_is_refused(self):
        # every directed pair among three cards IS a rig link, so the honest
        # negative is an out-of-range card, not a reversed one.
        with self.assertRaises(ValueError):
            xr.cross_sem_name("nonce", 0, 5, 0, "full")

    def test_the_resolved_cross_name_is_the_pair_name(self):
        self.assertEqual(xr.cross_sem_name("nonce", 1, 0, 0, "full"),
                         xr.sem_name("nonce", xr.CROSS_PAIRS.index((1, 0)),
                                     0, "full"))

    def test_w15_refusal_is_load_bearing(self):
        """THE DYING MUTANT: pull the guard, watch the test go red.

        This is the assert the guard exists for.  It re-runs the refusal with
        the guard's own predicate, so a future edit that silently turns
        ``src == dst`` into a legal cross name makes THIS test fail instead of
        the metal hanging.
        """
        refused = False
        try:
            xr.cross_sem_name("nonce", 1, 1, 0, "full")
        except xr.Weg2XchgDiagonalHasNoCrossPair:
            refused = True
        self.assertTrue(
            refused,
            "the W15 src==dst guard is gone: a diagonal id on the cross path "
            "resolves a foreign cross name again (the xsn52 hang)")


if __name__ == "__main__":
    unittest.main()
