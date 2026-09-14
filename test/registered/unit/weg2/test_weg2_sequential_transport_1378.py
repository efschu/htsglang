# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn44: DER SEQUENTIELLE EINHEITEN-TRANSPORT -- rot-first.

DAS DESIGN (im Record BOOT_weg2xsn38_0914.md, NUTZER + Coordinator
bestaetigt): EIN Host-Puffer (die groesste Einheit + Luft), die Einheiten
= die (tag, name)-Paare der PLAN-PARAM-Liste, der Deposit kopiert die
Einheit in den Puffer und postet EIN Signal, der Collect wartet auf das
Signal, vergleicht den Digest (BY NAME, Mismatch = Refusal) und kopiert
in sein Ziel, postet verbraucht. Sequenziell -- die Lane-Maschinerie
(Lanes, Permit, Cap, Band-Grenzen) ist dafuer nicht gebaut und
produzierte die elf Waende der Familie.

ROT-FIRST: die Tests unten pinnen den Kern (run_sequential_units) an
echten Semaphoren und einem echten shm-Puffer; vor der Implementierung
sind sie rot (die Funktion fehlte).

MUTANT (danger direction): das Mapping um EINE Einheit verschoben -- der
Collect konsumiert Einheit i, aber vergleicht/mit dem Digest von i+1 --
muss den Digest-Mismatch produzieren, nicht gruen durchlaufen.
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

    def memcpy_async(self, dst, src, nbytes, stream):
        if isinstance(dst, memoryview):
            dst[:nbytes] = bytes(src[:nbytes])
        elif isinstance(src, memoryview):
            self.vram[dst] = bytes(src[:nbytes])
        else:
            raise AssertionError(f"memcpy_async: unknown shape {dst!r}")

    def synchronize(self, stream=0):
        return None


def _units(nbytes_list):
    return [(f"unit{i}", f"weights_{i}", n, None, None)
            for i, n in enumerate(nbytes_list)]


class TheSequentialTransport(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-seq-")
        self.nonce = f"seqtest{os.getpid()}"
        # the launcher creates all 24 before either group starts; the test
        # plays the launcher for its own nonce (unlink first, then O_CREAT).
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
        """The happy path: the deposit's digest and the collect's digest
        agree per unit -- the transport moves the bytes."""
        ops = _FakeOps()
        units = _units([64, 128, 32])
        # the deposit: the src addrs point into the fake VRAM
        src_addrs = [0x1000 + i * 0x100 for i in range(len(units))]
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        rc = bx.run_sequential_units(
            deposit_units, ops, self.nonce, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        self.assertEqual(rc, "", f"the deposit must run clean: {rc}")

        # the collect: the dst addrs point into the fake VRAM (fresh)
        collect_units = [(u[0], u[1], u[2], None, 0x2000 + i * 0x100)
                         for i, u in enumerate(units)]
        rc = bx.run_sequential_units(
            collect_units, ops, self.nonce, shm_root=self.root,
            phase=bx.PHASE_COLLECT, digest_fn=None)
        self.assertEqual(rc, "", f"the collect must run clean: {rc}")

    def test_digest_mismatch_is_a_named_refusal(self):
        """The deposit's digest and the collect's read disagree -- the
        mismatch must be a named refusal, never a silent pass."""
        ops = _FakeOps()
        units = _units([64, 128])
        src_addrs = [0x1000, 0x1100]
        deposit_units = [(u[0], u[1], u[2], src_addrs[i], None)
                         for i, u in enumerate(units)]
        nonce2 = self.nonce + "b"
        xr.unlink_semaphores(nonce2)
        xr.create_semaphores(nonce2)
        bx.run_sequential_units(
            deposit_units, ops, nonce2, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, digest_fn=None)
        # THE MUTANT: the collect's unit list is ROTATED by one relative to
        # the deposit's -- its unit 0 carries unit1's name/size and reads
        # buf[:128] where the deposit posted unit0's 64 bytes. The digest
        # over the wrong window differs from the deposit's record.
        shifted = [(units[1][0], units[1][1], units[1][2], None, 0x2000),
                   (units[0][0], units[0][1], units[0][2], None, 0x2100)]
        rc = bx.run_sequential_units(
            shifted, ops, self.nonce + "b", shm_root=self.root,
            phase=bx.PHASE_COLLECT, digest_fn=None)
        # the shift must produce a NAMED refusal, not a silent pass
        self.assertIn("digest mismatch", rc,
                      f"the shifted mapping must die on the digest: {rc!r}")


if __name__ == "__main__":
    unittest.main()
