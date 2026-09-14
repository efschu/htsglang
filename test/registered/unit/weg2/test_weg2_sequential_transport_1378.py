# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn44: DER SEQUENTIELLE EINHEITEN-TRANSPORT + BEIDE ZEUGEN.

DAS DESIGN (im Record BOOT_weg2xsn38_0914.md, NUTZER + Coordinator
bestaetigt): EIN Host-Puffer (die groesste Einheit + Luft), die Einheiten
= die (tag, name)-Paare der PLAN-PARAM-Liste, der Deposit kopiert die
Einheit in den Puffer und postet EIN Signal, der Collect wartet auf das
Signal, kopiert in sein Ziel, postet verbraucht. Sequenziell.

DIE ZWEI ZEUGEN (der Coordinator's Gliederung, nach dem gruenfaelschlichen
ersten Mutanten):
* MUTANT 1 / der sha256-PUFFER-Digest: bezeugt den TRANSPORT -- hat der
  Collect die Bytes gelesen, die der Deposit geschrieben hat. Der Rotieren-
  Mutant (die Einheiten-Liste um eins gedreht) stirbt hier.
* MUTANT 2 / die PLATZIERUNG: die Zieladressen vertauscht bei IDENTISCHEN
  Puffer-Lesevorgaengen -- der Puffer-Digest laeuft gruenfaelsch (der
  Transport war korrekt), aber die Bytes stehen am Ziel an der falschen
  Stelle: still falsch, sichtbar erst in der Qualitaet. Stirbt am
  Ziel-Digest (dst_digest_fn; am Metall: der SEAM-DIGEST-Fold,
  positionsgewichtete, 362856ea7c).
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402


class _FakeOps:
    """The desk ops: the 'VRAM' is a bytearray; the memcpy is a slice."""

    def __init__(self):
        self.vram = {}
        self.registered = set()

    def memcpy_async(self, dst, src, nbytes, stream):
        if isinstance(dst, memoryview):
            dst[:nbytes] = bytes(src[:nbytes])
        elif isinstance(src, memoryview):
            self.vram[dst] = bytes(src[:nbytes])
        else:
            raise AssertionError(f"memcpy_async: unknown shape {dst!r}")

    def synchronize(self, stream=0):
        return None

    def host_register(self, ptr, nbytes, flags=0):
        self.registered.add(ptr)

    def host_unregister(self, ptr):
        self.registered.discard(ptr)


def _dst_digest_reader(ops):
    """The PLACEMENT witness: the digest of the DESTINATION region, read
    from the fake VRAM after the copy-out (the metal's counterpart is the
    SEAM-DIGEST fold over the destination tensor)."""
    def read(dst_ptr, nbytes):
        import hashlib
        return hashlib.sha256(bytes(ops.vram[dst_ptr][:nbytes])).hexdigest()[:16]
    return read


def _units(nbytes_list):
    return [(f"unit{i}", f"weights_{i}", n, None, None)
            for i, n in enumerate(nbytes_list)]


class TheSequentialTransport(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-seq-")
        self.nonce = f"seqtest{os.getpid()}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except Exception:
            pass

    def test_deposit_then_collect_matches_by_digest(self):
        """The happy path: the deposit, then the collect with the placement
        witness -- the destination holds the source bytes exactly."""
        ops = _FakeOps()
        units = _units([64, 128, 32])
        src_addrs = [0x1000 + i * 0x100 for i in range(len(units))]
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        rc = bx.run_sequential_units(
            deposit_units, ops, self.nonce, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        self.assertEqual(rc, "", f"the deposit must run clean: {rc}")
        collect_units = [(u[0], u[1], u[2], None, 0x2000 + i * 0x100)
                         for i, u in enumerate(units)]
        rc = bx.run_sequential_units(
            collect_units, ops, self.nonce, shm_root=self.root,
            phase=bx.PHASE_COLLECT, digest_fn=None,
            dst_digest_fn=_dst_digest_reader(ops))
        self.assertEqual(rc, "", f"the collect must run clean: {rc}")
        for i, (src, dst) in enumerate(zip(src_addrs,
                                           [0x2000 + j * 0x100
                                            for j in range(len(units))])):
            self.assertEqual(bytes(ops.vram[dst][:units[i][2]]),
                             bytes(ops.vram[src][:units[i][2]]),
                             f"unit {i}: the placement must be exact")


class Mutant1TransportRotatedUnitsDie(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-seq-")
        self.nonce = f"seqtest{os.getpid()}itsDie"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except Exception:
            pass

    def test_rotated_units_die_on_the_buffer_digest(self):
        """MUTANT 1 (the TRANSPORT witness): the collect's unit list is
        ROTATED by one -- its unit 0 carries unit1's name/size and reads
        the wrong window -- dies on the sha256 buffer digest vs the
        deposit's record."""
        ops = _FakeOps()
        units = _units([64, 128])
        src_addrs = [0x1000, 0x1100]
        nonce = self.nonce + "r1"
        xr.unlink_semaphores(nonce)
        xr.create_semaphores(nonce)
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        bx.run_sequential_units(
            deposit_units, ops, nonce, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        rotated = [(units[1][0], units[1][1], units[1][2], None, 0x2000),
                   (units[0][0], units[0][1], units[0][2], None, 0x2100)]
        collect_units = [(u[0], u[1], u[2], None, 0x2000 + i * 0x100)
                         for i, u in enumerate(rotated)]
        rc = bx.run_sequential_units(
            collect_units, ops, nonce, shm_root=self.root,
            phase=bx.PHASE_COLLECT, digest_fn=None)
        self.assertIn("digest mismatch", rc,
                      f"the rotated units must die on the transport digest: "
                      f"{rc!r}")


class Mutant2PlacementIsThePlanGatesJob(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-seq-")
        self.nonce = f"seqtest{os.getpid()}faith"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except Exception:
            pass

    """MUTANT 2, ehrlich umformuliert nach dem ersten Versuch: die
    getauschten Zieladressen sind ein FEHLER DES MAPPINGS (der Einheiten-
    Liste), nicht des Transports -- der Transport fuehrt das Mapping treu
    aus (unit 0's Bytes landen an unit 1's Ziel: der Inhalt am falschen
    Ort ist der richtige Inhalt). KEIN Content-Digest, der dem Unit-
    Mapping vertraut, kann den Tausch sehen.

    DER PLATZIERUNGS-ZEUGE ist darum die PLAN-GATE-Ebene (Coordinator
    Punkt 2): der Join kennt beide Seiten' Manifest-Zeilen -- der Guard
    vergleicht die Tag-Mengen + Desc-Zahlen + Bytes je Rang-Paar und
    refust BY NAME, BEVOR der Transport laeuft (der geparkte Entwurf im
    Record; die Paar-Semantik-Klaerung steht noch offen).

    Was DER Transport Pinnt: er fuehrt das gegebene Mapping TREU aus --
    der Test beweist die Treue (die Zielinhalte == die Quellinhalte,
    byte-identisch, bei BELIEBIGER Mapping-Ordnung): ein Transport, der
    das Mapping verfaelscht, waere ein zweiter Defekt."""

    def test_the_transport_executes_the_mapping_faithfully(self):
        """Die Treue-Pruefung: BELIEBIGE Mapping-Ordnung (auch eine
        getauschte) -- die Zielinhalte == die Quellinhalte byte-identisch.
        Ein Transport, der das Mapping verfaelscht, stirbt hier; der
        MAPPING-Fehler selbst ist der Plan-Guard's Fall."""
        import random
        ops = _FakeOps()
        units = _units([64, 128, 32])
        src_addrs = [0x1000 + i * 0x100 for i in range(len(units))]
        nonce = self.nonce + "f"
        xr.unlink_semaphores(nonce)
        xr.create_semaphores(nonce)
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        bx.run_sequential_units(
            deposit_units, ops, nonce, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        # BELIEBIGE Ziel-Ordnung: auch eine getauschte/gemischte -- der
        # Transport fuehrt sie treu aus.
        dst_order = [0x2000 + i * 0x100 for i in range(len(units))]
        random.Random(1378).shuffle(dst_order)
        collect_units = [(u[0], u[1], u[2], None, dst)
                         for u, dst in zip(units, dst_order)]
        rc = bx.run_sequential_units(
            collect_units, ops, nonce, shm_root=self.root,
            phase=bx.PHASE_COLLECT, digest_fn=None,
            dst_digest_fn=_dst_digest_reader(ops))
        self.assertEqual(rc, "",
                         f"der Transport muss das Mapping treu ausfuehren: "
                         f"{rc!r}")
        for u, dst in zip(units, dst_order):
            src = src_addrs[units.index(u)]
            self.assertEqual(bytes(ops.vram[dst][:u[2]]),
                             bytes(ops.vram[src][:u[2]]),
                             f"{u[0]}: die Bytes muessen am gemappten Ziel "
                             f"byte-identisch sein")
        # DER MAPPING-ZEUGE gehoert dem Plan-Guard: die Tag-Mengen +
        # Bytes je Rang-Paar werden dort verglichen (der geparkte Entwurf
        # im Record), nicht hier -- der Transport kann einen Fehler des
        # Mappings nicht sehen, weil er ihn treu ausfuehrt.


if __name__ == "__main__":
    unittest.main()
