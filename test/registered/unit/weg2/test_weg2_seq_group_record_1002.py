# SPDX-License-Identifier: Apache-2.0
"""FLIP-LEGS 02.10.: the on-card SEQ lane writes ONE record file per sync group.

MEASURED (N5a b49f0282c2 1002_114540, the slow D->P flip ep13): D TP1's on-card
deposit of weights_5 (lane c1, 114 units) took record_ms=475 against 8 ms on
every other flip -- the per-unit record files (open + json + tmp/rename for
each unit, every syscall a GIL release that must be won back behind the
rank's other threads). That deposit held TP1's whole tag 520 ms (746-1266 ms),
PP0's p2 collect of the same tag waited 495 ms for it, and the flip's legs
read 1742 ms against 1363 ms on the flip before. Across boots: SEQ deposit
record_ms max 135-563 ms (1-3 per boot), D collect record_ms p90 93-197 ms,
max 305-626 ms (sum 4-15 s per boot).

Pinned here (red before): the depositor writes one ``<dpath>.g<first unit>``
per sync group and no unit files; the collector reads one file per group
(lane-time ``records=group:N``) and moves the same bytes; a depositor with
the switch off writes unit files and the collector falls back to them; a
stale group file left by an earlier tag is never taken for this tag's record.
"""

from __future__ import annotations

import glob
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




def _descs(h, nbytes, tag="weights_0", base=0):
    out = []
    for i, n in enumerate(nbytes):
        src, dst = h._vram_pair(n, base + i)
        h.ops.write(src, bytes([(0x30 + base + i) & 0xFF]) * n)
        out.append(_desc(f"unit{base + i}", n, src_ptr=src, dst_ptr=dst))
    return out


class GroupRecords(_TransportHarness):
    def _env(self, **kv):
        saved = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in saved.items()])

    def _records(self, kind):
        return sorted(glob.glob(os.path.join(self.root, "**", f"*.{kind}*"), recursive=True))

    def test_switch_default_on(self):
        self.assertTrue(bx.seq_group_record_on({}))
        self.assertFalse(bx.seq_group_record_on({bx.ENV_SEQ_GROUP_RECORD: "0"}))

    def test_one_file_per_group_and_the_bytes_move(self):
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: None})
        descs = _descs(self, [64, 128, 32, 16])
        self.assertEqual(self._deposit(descs), "")
        units = [p for p in self._records("u") if not p.endswith(".tmp")]
        groups = [p for p in self._records("g") if not p.endswith(".tmp")]
        self.assertEqual(units, [], units)
        self.assertEqual(len(groups), 1, groups)       # 4 small units: one sync group
        self.assertEqual(self._collect(descs), "")
        for i, d in enumerate(descs):
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)),
                             bytes([0x30 + i]) * int(d.nbytes))
        col = [l for l in self.lines if "WEG2-SEQ lane-time" in l and "phase=collect" in l][-1]
        dep = [l for l in self.lines if "WEG2-SEQ lane-time" in l and "phase=deposit" in l][-1]
        self.assertIn("records=group:4", col)
        self.assertIn("records=group ", dep)

    def test_a_unit_file_depositor_is_still_read(self):
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: "0"})
        descs = _descs(self, [64, 32])
        self.assertEqual(self._deposit(descs), "")
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: None})
        self.assertEqual(self._collect(descs), "")
        col = [l for l in self.lines if "WEG2-SEQ lane-time" in l and "phase=collect" in l][-1]
        self.assertIn("records=group:0", col)
        for i, d in enumerate(descs):
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)),
                             bytes([0x30 + i]) * int(d.nbytes))

    def test_a_stale_group_file_is_never_this_tags_record(self):
        # tag A with group records, then tag B with unit files: B's collect
        # must not take A's group file for its units
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: None})
        a = _descs(self, [64, 32], tag="weights_0")
        self.assertEqual(self._deposit(a), "")
        self.assertEqual(self._collect(a), "")
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: "0"})
        b = _descs(self, [48, 16], tag="weights_0", base=10)
        self.assertEqual(self._deposit(b), "")
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: None})
        self.assertEqual(self._collect(b), "")
        for i, d in enumerate(b):
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)),
                             bytes([(0x30 + 10 + i) & 0xFF]) * int(d.nbytes))

    def test_identity_mismatch_still_dies_with_group_records(self):
        self._env(**{bx.ENV_SEQ_GROUP_RECORD: None})
        descs = _descs(self, [64, 64])
        self.assertEqual(self._deposit(descs), "")
        wrong = [_desc("other0", 64, src_ptr=descs[0].src_ptr, dst_ptr=descs[0].dst_ptr),
                 descs[1]]
        rc = self._collect(wrong)
        self.assertTrue("identity mismatch" in rc or "record missing" in rc, rc)


if __name__ == "__main__":
    unittest.main()
