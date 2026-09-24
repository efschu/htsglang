# SPDX-License-Identifier: Apache-2.0
"""H46: die Ringdateien der On-card-Host-Lanes (H44) werden beim Boot
angelegt, befuellt und registriert -- nicht beim ersten Host-Weg-Tag im Flip.

Befund (x148 P+D-Log): alle new/grow-Registrierungen des Boots fielen in den
ersten D->P-Flip, cudaHostRegister derselben Groesse brauchte dort 5,9-7,9 s
(x147/x150: 293-342 ms), D-Deposits warteten 6-8,4 s auf ihre Abholer.

Die Tests fahren den ECHTEN Ring aus H44 (echte POSIX-Semaphoren, echte
mmaps ueber ``_persistent_host_buffer``, Depositor und Collector in zwei
Threads) nach dem Vorab-Register und halten fest:
* jede Ringdatei, die ein Flip dieses Rangs anlegen kann (je Pufferslot),
  entsteht beim Boot in genau der Groesse des Flips, einmal registriert,
  mit der Zeile ``WEG2-SEQ persist lane=c<k> preregister size= register_ms=``;
* der Flip danach registriert NICHTS mehr: nur ``persist ... reuse
  file=c<k>[_s1]_ring.bin``, kein new/grow, keine weitere host_register;
* ohne Vorab-Register registriert der erste Host-Tag im Flip (Kontrolle);
* die Gates (Ring aus, Persist aus, Freigabe je Leg, Schalter aus) legen
  nichts an; nur Diagonal-Lanes;
* der Boot-Thread bindet das Geraet des Rangs VOR dem Register.
Mutanten: siehe ``RingPreregisterMutants``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

MIB = 1 << 20
CARD = 1


class _Ops:
    """Host-Stellvertreter fuer VRAM mit einem ZAEHLENDEN host_register."""

    def __init__(self):
        self.real, self.size, self.keep = {}, {}, {}
        self.registered = []   # (addr, nbytes, flags)
        self.devices = []
        self.fail_register = False

    # -- the calls under test ---------------------------------------------
    def host_register(self, addr, nbytes, flags):
        if self.fail_register:
            raise RuntimeError("cudaHostRegister: out of memory (simulated)")
        self.registered.append((int(addr), int(nbytes), int(flags)))

    def host_unregister(self, addr):
        return None

    def set_device(self, dev):
        self.devices.append(int(dev))

    # -- fake VRAM ---------------------------------------------------------
    def hook(self, addr, nbytes):
        buf = ctypes.create_string_buffer(nbytes)
        self.keep[addr] = buf
        self.real[addr] = ctypes.addressof(buf)
        self.size[addr] = nbytes

    def _r(self, addr):
        for fake, real in self.real.items():
            if fake <= addr < fake + self.size[fake]:
                return real + (addr - fake)
        return addr

    def write(self, addr, data):
        ctypes.memmove(self._r(addr), bytes(data), len(data))

    def read(self, addr, nbytes):
        return ctypes.string_at(self._r(addr), nbytes)

    def digest(self, addr, nbytes):
        return hashlib.sha256(self.read(addr, nbytes)).hexdigest()[:16]

    def memcpy_async(self, dst, src, nbytes, stream):
        ctypes.memmove(self._r(dst), self._r(src), nbytes)

    def memcpy2d_async(self, dst, dpitch, src, spitch, run, rows, stream):
        for r in range(rows):
            self.memcpy_async(dst + r * dpitch, src + r * spitch, run, stream)

    def synchronize(self, stream=0):
        return None


def _desc(name, nbytes, src_ptr, dst_ptr, tag):
    return wx.XchgDesc(
        tag=tag, src_rank=CARD, dst_rank=CARD, param_name=name, kind=tp.FLAT,
        nbytes=nbytes, rows=1, run_bytes=nbytes, spitch=0, dpitch=0,
        src_ptr=src_ptr, dst_ptr=dst_ptr)


class _Base(unittest.TestCase):
    SLOTS = 2
    DEPTH = 2
    SIZES = [512 * 1024] * 6 + [int(1.5 * MIB)]

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-ring-h46-")
        self.nonce = f"h46r{os.getpid()}{int(time.time() * 1000) % 100000}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)
        self._env_before = {k: os.environ.get(k) for k in (
            bx.SEQ_SYNC_BATCH_MIB_ENV, bx.SEQ_UNIT_DIGEST_ENV,
            bx.SEQ_BUFFER_DEPTH_ENV, bx.SEQ_PERSIST_BUFFERS_ENV,
            bx.SEQ_RELEASE_LANES_ENV, bx.SEQ_HOST_REGISTER_ENV)}
        os.environ[bx.SEQ_SYNC_BATCH_MIB_ENV] = "1"
        os.environ[bx.SEQ_UNIT_DIGEST_ENV] = "1"
        os.environ[bx.SEQ_BUFFER_DEPTH_ENV] = str(self.DEPTH)
        os.environ.pop(bx.SEQ_PERSIST_BUFFERS_ENV, None)
        os.environ.pop(bx.SEQ_RELEASE_LANES_ENV, None)
        os.environ.pop(bx.SEQ_HOST_REGISTER_ENV, None)
        self._ov = [envs.SGLANG_WEG2_SEQ_LANE_RING_SLOTS.override(self.SLOTS),
                    envs.SGLANG_WEG2_SEQ_LANE_RING_READY_MS.override(3000)]
        for o in self._ov:
            o.__enter__()
        self.boot, self.dlines, self.clines = [], [], []

    def tearDown(self):
        for o in reversed(self._ov):
            o.__exit__(None, None, None)
        for k, v in self._env_before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        bx.release_host_lane_buffers(truncate=False, log=lambda *_a: None)
        bx.release_stage_buffers(self.nonce)
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            xr.unlink_semaphores(self.nonce)
        except BaseException:
            pass

    def _forget_process_cache(self):
        """A second process (the co-card peer) has its own cache: drop this
        process's entries without touching the files."""
        with bx._SEQ_CACHE_LOCK:
            for p in [p for p in bx._SEQ_HOST_BUF if p.startswith(self.root)]:
                ent = bx._SEQ_HOST_BUF.pop(p)
                try:
                    ent["mm"].close()
                    ent["fh"].close()
                except Exception:
                    pass

    def _prereg(self, ops, lanes=(f"c{CARD}",), depth=None):
        return bx.preregister_ring_lanes(self.nonce, list(lanes), ops,
                                         log=self.boot.append,
                                         shm_root=self.root, depth=depth)

    def _lane_dir(self):
        return os.path.dirname(bx.seq_ring_path(self.nonce, self.root, lane="c1"))

    def _ring_bytes(self):
        return bx.seq_ring_bytes(self.SLOTS, MIB)

    # -- one tag through the REAL H44 ring (two threads) --------------------
    def _run(self, descs, ops, phase, lines, **kw):
        return bx.run_sequential_units(
            descs, ops, self.nonce, slot_bytes=1 << 30, shm_root=self.root,
            phase=phase, card=CARD, log=lines.append, **kw)

    def _round_trip(self, dops, cops, *, tag, buffer_slot):
        descs, want = [], []
        for i, n in enumerate(self.SIZES):
            src = 0x1000_0000 + i * 0x0100_0000 + buffer_slot * 0x1000_0000_0
            dst = 0x4000_0000 + i * 0x0100_0000 + buffer_slot * 0x1000_0000_0
            dops.hook(src, n)
            cops.hook(dst, n)
            data = (hashlib.sha256(f"{tag}/{i}".encode()).digest() * (n // 32 + 1))[:n]
            dops.write(src, data)
            descs.append(_desc(f"u{i}", n, src, dst, tag))
            want.append((dst, data))
        out = {}

        def body():
            out["rc"] = self._run(descs, cops, bx.PHASE_COLLECT, self.clines,
                                  dst_digest_fn=cops.digest, buffer_slot=buffer_slot)

        t = threading.Thread(target=body, daemon=True)
        t.start()
        ready = bx.seq_ready_path(bx.sequential_digest_path(
            self.nonce, self.root, lane=bx.seq_lane_file_name(f"c{CARD}", buffer_slot)))
        t0 = time.time()
        while not os.path.exists(ready):
            if time.time() - t0 > 10:
                self.fail("the collector never announced readiness")
            time.sleep(0.002)
        rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines,
                       buffer_slot=buffer_slot)
        t.join(60)
        self.assertFalse(t.is_alive(), "collector hung")
        self.assertEqual((rc, out.get("rc")), ("", ""), (self.dlines[-3:], self.clines[-3:]))
        for dst, data in want:
            self.assertEqual(cops.read(dst, len(data)), data)

    def _flip(self, ops_d, ops_c):
        """Both buffer slots of the lane take the ring once (depth 2)."""
        for slot in range(self.DEPTH):
            self._round_trip(ops_d, ops_c, tag=f"weights_{3 + slot}", buffer_slot=slot)


class RingPreregister(_Base):
    def test_every_ring_file_of_the_rank_exists_registered_at_boot(self):
        ops = _Ops()
        rec = self._prereg(ops)
        want = sorted(["c1_ring.bin", "c1_s1_ring.bin"])
        self.assertEqual(sorted(os.listdir(self._lane_dir())), want)
        for name in want:
            self.assertEqual(os.path.getsize(os.path.join(self._lane_dir(), name)),
                             self._ring_bytes())
        self.assertEqual([(n, f) for _a, n, f in ops.registered],
                         [(self._ring_bytes(), tp.CUDA_HOST_REGISTER_PORTABLE)] * 2)
        self.assertEqual((rec["files"], rec["registered"], rec["refused"]), (2, 2, 0))
        self.assertEqual(rec["bytes"], 2 * self._ring_bytes())
        lines = [ln for ln in self.boot if "WEG2-SEQ persist lane=c1 preregister" in ln]
        self.assertEqual(len(lines), 2, self.boot)
        for ln, name in zip(lines, want):
            self.assertIn(f"preregister size={self._ring_bytes()} ", ln)
            self.assertIn("registered=yes register_ms=", ln)
            self.assertIn("populate_ms=", ln)
            self.assertTrue(ln.endswith(f"file={name}"), ln)
        self.assertTrue(any("WEG2-SEQ preregister done lanes=c1 files=2/2" in ln
                            for ln in self.boot), self.boot)
        # the H44 host census books the boot-time files like the flip's
        self.assertTrue(any("WEG2-HOST-LEDGER LANES" in ln and "file=c1_ring.bin" in ln
                            for ln in self.boot), self.boot)

    def test_the_flip_after_preregister_only_reuses_and_registers_nothing(self):
        ops_d, ops_c = _Ops(), _Ops()
        self._prereg(ops_d)
        n_boot = len(ops_d.registered)
        self._flip(ops_d, ops_c)
        flip = self.dlines + self.clines
        persist = [ln for ln in flip if "WEG2-SEQ persist lane=c1" in ln]
        self.assertTrue(persist, flip)
        self.assertEqual([ln for ln in persist if " reuse " not in ln], [], persist)
        self.assertEqual({ln.rsplit("file=", 1)[1] for ln in persist},
                         {"c1_ring.bin", "c1_s1_ring.bin"})
        self.assertEqual(len(ops_d.registered), n_boot, "the flip registered again")
        self.assertEqual(ops_c.registered, [], "the collector registered in the flip")
        self.assertTrue(any("host=ring why=collector-ready" in ln for ln in self.dlines))

    def test_control_without_preregister_the_first_host_tag_registers_in_the_flip(self):
        ops_d, ops_c = _Ops(), _Ops()
        self._flip(ops_d, ops_c)
        flip = self.dlines + self.clines
        new = [ln for ln in flip if "WEG2-SEQ persist lane=c1 new" in ln]
        self.assertEqual(len(new), 2, flip)  # one per buffer slot, inside the flip
        self.assertEqual(len(ops_d.registered) + len(ops_c.registered), 2)

    def test_the_co_card_peer_maps_the_same_files_without_growing_them(self):
        ops_p, ops_d = _Ops(), _Ops()
        self._prereg(ops_p)
        sizes = {n: os.path.getsize(os.path.join(self._lane_dir(), n))
                 for n in os.listdir(self._lane_dir())}
        self._forget_process_cache()          # the other process's own cache
        self._prereg(ops_d)
        self.assertEqual({n: os.path.getsize(os.path.join(self._lane_dir(), n))
                          for n in os.listdir(self._lane_dir())}, sizes)
        self.assertEqual(len(ops_d.registered), 2, "each process pins its own mapping")

    def test_depth_one_is_one_file(self):
        ops = _Ops()
        rec = self._prereg(ops, depth=1)
        self.assertEqual(os.listdir(self._lane_dir()), ["c1_ring.bin"])
        self.assertEqual(rec["files"], 1)

    def test_only_diagonal_lanes(self):
        ops = _Ops()
        rec = self._prereg(ops, lanes=("p0", "p3", "c1", "x", "c"))
        self.assertEqual(rec["files"], 2)
        persist = [ln for ln in self.boot if ln.startswith("WEG2-SEQ persist ")]
        self.assertEqual(len(persist), 2, self.boot)
        self.assertTrue(all(ln.startswith("WEG2-SEQ persist lane=c1 ") for ln in persist), persist)
        self.assertIn("WEG2-SEQ preregister done lanes=c1 ", self.boot[-1])

    def test_a_refused_register_is_named_and_left_to_the_flip(self):
        ops = _Ops()
        ops.fail_register = True
        rec = self._prereg(ops)
        self.assertEqual((rec["files"], rec["refused"]), (0, 2))
        self.assertTrue(any("WEG2-SEQ preregister lane=c1 file=c1_ring.bin REFUSED" in ln
                            for ln in self.boot), self.boot)
        with bx._SEQ_CACHE_LOCK:
            self.assertFalse(any(p.startswith(self.root) for p in bx._SEQ_HOST_BUF),
                             "a refused file must not be cached as registered")


class RingPreregisterGates(_Base):
    def _assert_nothing(self, rec, why_part):
        self.assertEqual(rec["files"], 0)
        self.assertIn(why_part, rec["skipped"])
        self.assertFalse(os.path.exists(self._lane_dir()))
        self.assertTrue(any("WEG2-SEQ preregister skipped" in ln for ln in self.boot))

    def test_switch_off(self):
        with envs.SGLANG_WEG2_SEQ_LANE_RING_PREREGISTER.override(False):
            self._assert_nothing(self._prereg(_Ops()), "PREREGISTER=0")

    def test_ring_off(self):
        with envs.SGLANG_WEG2_SEQ_LANE_RING.override(False):
            self._assert_nothing(self._prereg(_Ops()), "SGLANG_WEG2_SEQ_LANE_RING=0")

    def test_persist_off(self):
        os.environ[bx.SEQ_PERSIST_BUFFERS_ENV] = "0"
        self._assert_nothing(self._prereg(_Ops()), bx.SEQ_PERSIST_BUFFERS_ENV)

    def test_release_per_leg(self):
        os.environ[bx.SEQ_RELEASE_LANES_ENV] = "1"
        self._assert_nothing(self._prereg(_Ops()), bx.SEQ_RELEASE_LANES_ENV)

    def test_the_default_is_on(self):
        self.assertTrue(envs.SGLANG_WEG2_SEQ_LANE_RING_PREREGISTER.get())
        self.assertEqual(bx.ring_preregister_skip_reason(), "")


# --------------------------------------------------------------------------
# the rank side: the boot thread
# --------------------------------------------------------------------------

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

Manager = wu.SchedulerWeightUpdaterManager


class _FakeRunner:
    model = None
    model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _manager(rank=2):
    m = Manager(tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
                memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
                is_fully_idle=lambda *a, **k: True)
    return m


class RankThread(_Base):
    def test_the_thread_binds_the_rank_device_before_it_registers(self):
        m = _manager()
        order = []
        ops = _Ops()
        _reg = ops.host_register
        ops.host_register = lambda *a: (order.append("register"), _reg(*a))
        orig = bx.preregister_ring_lanes
        with mock.patch.object(bx, "preregister_ring_lanes",
                               wraps=lambda nonce, lanes, o, log: orig(
                                   nonce, lanes, o, log=log, shm_root=self.root)) as pr:
            rec = m._weg2_ring_preregister(
                boot_nonce=self.nonce, rank=CARD, device=2, ops=ops,
                set_device=lambda d: order.append(f"device{d}"))
        self.assertEqual(order[0], "device2", order)
        self.assertEqual(order[1:], ["register", "register"])
        self.assertEqual(pr.call_args.args[1], [f"c{CARD}"])
        self.assertEqual(rec["files"], 2)

    def test_no_device_means_no_register(self):
        m = _manager()
        ops = _Ops()
        rec = m._weg2_ring_preregister(boot_nonce=self.nonce, rank=CARD, device=-1,
                                       ops=ops, set_device=lambda d: None)
        self.assertEqual(rec, {})
        self.assertEqual(ops.registered, [])

    def test_start_wires_the_thread_at_boot(self):
        m = _manager()
        from sglang.srt.weg2 import weight_exchange as _wx
        seen = threading.Event()
        with mock.patch.object(_wx, "exchange_armed", return_value=True), \
                mock.patch.object(Manager, "_weg2_bar1_start", lambda self: None), \
                mock.patch.object(Manager, "_weg2_join_prewarm_start", lambda self: None), \
                mock.patch.object(Manager, "_weg2_rank", lambda self: CARD), \
                mock.patch.object(Manager, "_weg2_ring_preregister",
                                  lambda self, **kw: seen.set()), \
                mock.patch.dict(os.environ, {xr.ENV_REGION_BOOT: self.nonce}):
            m._weg2_prewarm_lanes_start()
            self.assertIsNotNone(m._weg2_ring_prereg_thread)
            m._weg2_ring_prereg_thread.join(10)
        self.assertTrue(seen.is_set())

    def test_start_is_a_no_op_with_the_switch_off(self):
        m = _manager()
        with envs.SGLANG_WEG2_SEQ_LANE_RING_PREREGISTER.override(False), \
                mock.patch.object(Manager, "_weg2_rank", lambda self: CARD), \
                mock.patch.dict(os.environ, {xr.ENV_REGION_BOOT: self.nonce}):
            m._weg2_ring_preregister_start()
        self.assertIsNone(m._weg2_ring_prereg_thread)


if __name__ == "__main__":
    unittest.main()
