# SPDX-License-Identifier: Apache-2.0
"""PD-LEGS-1024: the P->D weight legs' per-unit GIL round trips.

MEASURED (boot dkr27bint4arbar1int4ar10240037, fe572f3224, 2x RTX 3080, D TP0,
102 P->D tags of lane c0): the on-card SEQ collect of one weight tag (224 units,
787 MB) spends p50 66 / p90 158 ms OUTSIDE its waits (wait_ms p50 4 -- the
depositor is ahead, the tokens are posted) for 4 ms of copies. Per unit it made
one ``sem_timedwait`` and two ``cuPointerGetAttribute`` through ctypes CDLL --
three GIL releases, each won back behind the rank's BAR1 collector thread, which
pays the same contention back (27b-lanes3x80-1023: T per batch 1.75 ms with the
c-lane collect overlapping vs 0.75 ms without).

Pinned here (red on 47471fea4a):
* ``SemSet.trywait_held`` / ``diagonal_trywait_held`` exist, keep the GIL
  (PyDLL) and take EXACTLY one token when one is there, none when none is.
* the SEQ collect takes posted tokens GIL-held (lane-time ``held_takes=N/N``),
  moves the same bytes, leaves the semaphore at 0 (every posted token consumed
  exactly once -- the danger direction: a collect that copies a unit whose
  deposit has not posted, or that leaves/steals a token and so lets the NEXT
  tag's unit be read early = wrong weights after the flip).
* a collect that starts BEFORE its deposit still blocks: nothing is copied
  until the deposit posts.
* switch off = the blocking wait for every unit (``held_takes=0/N``).
* ``ptr_attrs`` binds ``cuPointerGetAttribute`` through PyDLL (switch on) or
  CDLL (off), and an unmapped destination is still refused by name.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import threading
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import weight_exchange as wx  # noqa: E402
from flliper.srt.pdflip import weight_exchange_bounce as bx  # noqa: E402
from flliper.srt.pdflip import weight_exchange_region as xr  # noqa: E402
from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402


class _MemOps:
    """Address-honest host stand-in for device memory (no CUDA)."""

    def __init__(self):
        self.bufs, self.real, self.size = {}, {}, {}

    def hook(self, addr, nbytes):
        buf = ctypes.create_string_buffer(nbytes)
        self.bufs[addr], self.real[addr], self.size[addr] = buf, ctypes.addressof(buf), nbytes

    def _real(self, addr):
        for fake, real in self.real.items():
            if fake <= addr < fake + self.size[fake]:
                return real + (addr - fake)
        return addr

    def write(self, addr, data):
        ctypes.memmove(self._real(addr), bytes(data), len(data))

    def read(self, addr, nbytes):
        return ctypes.string_at(self._real(addr), nbytes)

    def memcpy_async(self, dst, src, nbytes, stream):
        ctypes.memmove(self._real(dst), self._real(src), nbytes)

    def memcpy2d_async(self, dst, dpitch, src, spitch, run_bytes, rows, stream):
        for r in range(rows):
            ctypes.memmove(self._real(dst + r * dpitch), self._real(src + r * spitch), run_bytes)

    def synchronize(self, stream=0):
        return None


def _desc(name, nbytes, src_ptr, dst_ptr):
    return wx.XchgDesc(tag="weights_0", src_rank=1, dst_rank=1, param_name=name,
                       kind=tp.FLAT, nbytes=nbytes, rows=1, run_bytes=nbytes,
                       spitch=0, dpitch=0, src_ptr=src_ptr, dst_ptr=dst_ptr)


class _Env:
    def _env(self, **kv):
        saved = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        def _restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(_restore)


class _Sems(unittest.TestCase, _Env):
    def setUp(self):
        self.nonce = f"pdlegs{os.getpid()}{int(time.time() * 1000) % 100000}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)
        self.addCleanup(lambda: xr.unlink_semaphores(self.nonce))
        self.sems = tp.SemSet(self.nonce)


class HeldTrywait(_Sems):
    def test_held_entry_is_pydll(self):
        self.assertIsInstance(self.sems._held_libc(), ctypes.PyDLL)

    def test_diagonal_takes_exactly_one_token(self):
        slot = bx._SEQ_SLOT
        self.assertFalse(self.sems.diagonal_trywait_held(1, slot, "full"))
        self.sems.diagonal_post(1, slot, "full")
        self.sems.diagonal_post(1, slot, "full")
        self.assertTrue(self.sems.diagonal_trywait_held(1, slot, "full"))
        self.assertTrue(self.sems.diagonal_trywait_held(1, slot, "full"))
        self.assertFalse(self.sems.diagonal_trywait_held(1, slot, "full"))
        # nothing else was taken: a later post is available to a blocking wait
        self.sems.diagonal_post(1, slot, "full")
        self.assertTrue(self.sems.diagonal_timedwait(1, slot, "full", 0.5))
        self.assertFalse(self.sems.diagonal_timedwait(1, slot, "full", 0.05))

    def test_cross_takes_exactly_one_token(self):
        slot = bx._SEQ_SLOT
        self.assertFalse(self.sems.trywait_held(0, slot, "full"))
        self.sems.post(0, slot, "full")
        self.assertEqual(self.sems.getvalue(0, slot, "full"), 1)
        self.assertTrue(self.sems.trywait_held(0, slot, "full"))
        self.assertEqual(self.sems.getvalue(0, slot, "full"), 0)
        self.assertFalse(self.sems.trywait_held(0, slot, "full"))

    def test_take_helper_and_switch(self):
        slot = bx._SEQ_SLOT
        self.assertTrue(bx.seq_trywait_held_on({}))
        self.assertFalse(bx.seq_trywait_held_on({bx.ENV_SEQ_TRYWAIT_HELD: "0"}))
        self.sems.diagonal_post(2, slot, "full")
        self._env(**{bx.ENV_SEQ_TRYWAIT_HELD: "0"})
        self.assertFalse(bx.seq_take_held(self.sems, is_diagonal=True, card=2, pair=None, slot=slot))
        self._env(**{bx.ENV_SEQ_TRYWAIT_HELD: None})
        self.assertTrue(bx.seq_take_held(self.sems, is_diagonal=True, card=2, pair=None, slot=slot))
        self.assertFalse(bx.seq_take_held(self.sems, is_diagonal=True, card=2, pair=None, slot=slot))
        # a semaphore set without the held entry point (desk fakes): no take
        self.assertFalse(bx.seq_take_held(object(), is_diagonal=True, card=2, pair=None, slot=slot))

        class _Truthy:          # a duck-typed stand-in answering a truthy non-bool
            def diagonal_trywait_held(self, *a):
                return object()
        self.assertFalse(bx.seq_take_held(_Truthy(), is_diagonal=True, card=2, pair=None, slot=slot))


class SeqCollect(_Sems):
    N = 6

    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp(prefix="pdflip-pdlegs-")
        import shutil
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        # the whole-tag form (the path the metal's on-card IPC lane runs); a
        # collect that stands ready first would otherwise make the desk's host
        # depositor choose the H44 ring, a different path with its own waits
        self._env(FLLIPER_PDFLIP_SEQ_LANE_RING="0")
        self.ops = _MemOps()
        self.lines = []
        self.descs = []
        for i, n in enumerate([64, 128, 32, 16, 256, 48][: self.N]):
            src, dst = 0x10000 + i * 4096, 0x80000 + i * 4096
            self.ops.hook(src, n)
            self.ops.hook(dst, n)
            self.ops.write(src, bytes([0x41 + i]) * n)
            self.descs.append(_desc(f"unit{i}", n, src, dst))

    def _run(self, phase, budget_s=30.0):
        return bx.run_sequential_units(self.descs, self.ops, self.nonce, slot_bytes=1 << 30,
                                       shm_root=self.root, phase=phase, log=self.lines.append,
                                       card=1, budget_s=budget_s)

    def _col_line(self):
        return [l for l in self.lines if "PDFLIP-SEQ lane-time" in l and "phase=collect" in l][-1]

    def _bytes_ok(self):
        for i, d in enumerate(self.descs):
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)),
                             bytes([0x41 + i]) * int(d.nbytes), f"unit {i}")

    def _sem_left(self):
        v = ctypes.c_int(0)
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        lib.sem_getvalue.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        h = self.sems.diagonal_handle(1, bx._SEQ_SLOT, "full")
        self.assertEqual(lib.sem_getvalue(ctypes.c_void_p(h), ctypes.byref(v)), 0)
        return int(v.value)

    def test_posted_tokens_are_taken_held_and_the_bytes_move(self):
        self._env(**{bx.ENV_SEQ_TRYWAIT_HELD: None})
        self.assertEqual(self._run(bx.PHASE_DEPOSIT), "")
        self.assertEqual(self._sem_left(), self.N)
        self.assertEqual(self._run(bx.PHASE_COLLECT), "")
        self._bytes_ok()
        self.assertIn(f"held_takes={self.N}/{self.N} ", self._col_line())
        # every posted token consumed exactly once: none left for the next tag
        self.assertEqual(self._sem_left(), 0)

    def test_switch_off_is_the_blocking_form(self):
        self._env(**{bx.ENV_SEQ_TRYWAIT_HELD: "0"})
        self.assertEqual(self._run(bx.PHASE_DEPOSIT), "")
        self.assertEqual(self._run(bx.PHASE_COLLECT), "")
        self._bytes_ok()
        self.assertIn(f"held_takes=0/{self.N} ", self._col_line())
        self.assertEqual(self._sem_left(), 0)

    def test_a_collect_before_its_deposit_still_blocks(self):
        self._env(**{bx.ENV_SEQ_TRYWAIT_HELD: None})
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("rc", self._run(bx.PHASE_COLLECT)))
        t.start()
        time.sleep(0.4)
        self.assertTrue(t.is_alive(), "the collect returned before any deposit posted")
        for d in self.descs:
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)), b"\0" * int(d.nbytes),
                             "a destination was written before its deposit posted")
        self.assertEqual(self._run(bx.PHASE_DEPOSIT), "")
        t.join(30)
        self.assertFalse(t.is_alive())
        self.assertEqual(out.get("rc"), "")
        self._bytes_ok()
        self.assertEqual(self._sem_left(), 0)
        held = int(self._col_line().split("held_takes=")[1].split("/")[0])
        self.assertLessEqual(held, self.N)  # how many were held depends on the race; the bytes do not

    def test_a_held_miss_without_deposit_expires_by_name(self):
        self._env(**{bx.ENV_SEQ_TRYWAIT_HELD: None})
        rc = self._run(bx.PHASE_COLLECT, budget_s=0.5)
        self.assertTrue(rc.startswith("budget expired at unit 0"), rc)
        for d in self.descs:
            self.assertEqual(self.ops.read(int(d.dst_ptr), int(d.nbytes)), b"\0" * int(d.nbytes))
        self.assertEqual(self._sem_left(), 0)


class PtrAttrsHeld(unittest.TestCase, _Env):
    def _bind(self, monkeypatch_env):
        calls = []

        class _Fn:
            restype = None
            argtypes = None

            def __call__(self, out, attr, addr):
                return 1     # CUDA_ERROR_INVALID_VALUE: an address no context knows

        class _Lib:
            def __init__(self, kind):
                self.cuPointerGetAttribute = _Fn()
                calls.append(kind)

        saved = (bx._PTR_ATTRS_BINDING, ctypes.PyDLL, ctypes.CDLL)
        self._env(**{bx.ENV_PTRATTR_HELD: monkeypatch_env})
        try:
            bx._PTR_ATTRS_BINDING = None
            ctypes.PyDLL = lambda name, *a, **k: _Lib(("PyDLL", name))
            ctypes.CDLL = lambda name, *a, **k: _Lib(("CDLL", name))
            bx._ptr_attrs_binding()
            got = bx.ptr_attrs(0xDEAD0000)
        finally:
            bx._PTR_ATTRS_BINDING, ctypes.PyDLL, ctypes.CDLL = saved
            bx._PTR_ATTRS_BINDING = None
        return calls, got

    def test_switch_default_on(self):
        self.assertTrue(bx.ptrattr_held_on({}))
        self.assertFalse(bx.ptrattr_held_on({bx.ENV_PTRATTR_HELD: "off"}))

    def test_bound_through_pydll_by_default(self):
        calls, got = self._bind(None)
        self.assertEqual(calls, [("PyDLL", "libcuda.so.1")])
        self.assertEqual(got, (1, 0, -1))

    def test_off_is_the_cdll_binding(self):
        calls, got = self._bind("0")
        self.assertEqual(calls, [("CDLL", "libcuda.so.1")])
        self.assertEqual(got, (1, 0, -1))

    def test_unmapped_destination_is_still_refused(self):
        why = tp.refuse_unmapped_dst(lambda a: (1, 0, -1), set(), dst=0xDEAD0000,
                                     lane_key="c0", i=3, name="w", tag="weights_4")
        self.assertIn("is not mapped device memory", why)
        self.assertEqual(tp.refuse_unmapped_dst(lambda a: (0, 2, 0), set(), dst=0xBEEF0000,
                                                lane_key="c0", i=3, name="w", tag="weights_4"), "")


if __name__ == "__main__":
    unittest.main()
