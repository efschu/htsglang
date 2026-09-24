# SPDX-License-Identifier: Apache-2.0
"""H44 (Task #17): der Lane-Ring der On-card-Host-Lanes.

Die Tests fahren den ECHTEN Ring: echte benannte POSIX-Semaphoren
(``create_semaphores``), echte tmpfs-/tmp-mmaps ueber
``_persistent_host_buffer``, Depositor und Collector in zwei Threads bzw.
zwei Prozessen, Bytes per ``ctypes.memmove`` zwischen echt allozierten
Stellvertretern von VRAM. Keine GPU.

Was sie festhalten:
* Bytes bitgleich je Tag, auch mit mehr Batches als Slots (Wrap) und mit
  einer Einheit, die groesser als ein Slot ist (geschnitten).
* Collector langsamer als Depositor (der Depositor wartet auf Freigaben)
  und umgekehrt (der Collector wartet auf Tokens).
* Collector nicht bereit -> Ganz-Tag-Form, der Deposit endet OHNE Collector
  (#1374 Option 1, die Deadlock-Freiheit der Diagonale).
* Bereitschaft angezeigt, aber niemand leert -> W146 benannt, kein Stall.
* IPC-Tag mit Ring an: KEIN Host-Lane-File entsteht (die 3,2 GiB von x148).
* Schalter aus = alter Pfad (eager, keine host=-Felder, keine Ready-Datei).
* Mutanten: Freigabe vor dem Sync, Depositor ignoriert die Freigabe,
  Ready-Gate immer wahr -- jeder faellt.
"""

from __future__ import annotations

import ctypes
import hashlib
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import host_ledger as hl  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

MIB = 1 << 20
CARD = 1


class _Ops:
    """Adressehrliche Host-Stellvertreter fuer VRAM (je Seite eine Instanz).
    Unbekannte Adressen sind die geteilte mmap selbst."""

    def __init__(self, copy_delay_s: float = 0.0, deferred: bool = False):
        self.mem = {}
        self.real = {}
        self.size = {}
        self.copy_delay_s = float(copy_delay_s)
        self.deferred = bool(deferred)
        self.pending = []
        self.flush_on_sync = True
        self.syncs = 0

    def hook(self, addr, nbytes):
        buf = ctypes.create_string_buffer(nbytes)
        self.mem[addr] = buf
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

    def _do(self, dst, src, n):
        if self.copy_delay_s:
            time.sleep(self.copy_delay_s)
        ctypes.memmove(self._r(dst), self._r(src), n)

    def memcpy_async(self, dst, src, nbytes, stream):
        if self.deferred:
            self.pending.append((dst, src, nbytes))
        else:
            self._do(dst, src, nbytes)

    def memcpy2d_async(self, dst, dpitch, src, spitch, run, rows, stream):
        for r in range(rows):
            self.memcpy_async(dst + r * dpitch, src + r * spitch, run, stream)

    def synchronize(self, stream=0):
        self.syncs += 1
        if self.flush_on_sync:
            self.flush()

    def flush(self):
        ops, self.pending = self.pending, []
        for (d, s, n) in ops:
            self._do(d, s, n)


class _IpcOps(_Ops):
    """Mit On-card-IPC (die Produktform von 151/152 Transfers in x148)."""

    def __init__(self):
        super().__init__()
        self._next = 0x7000_0000

    def raw_malloc(self, device, nbytes):
        addr = self._next
        self._next += nbytes + 4096
        self.hook(addr, nbytes)
        return addr

    def raw_free(self, ptr):
        return None

    def ipc_get_handle(self, ptr):
        return int(ptr).to_bytes(8, "little")

    def ipc_open_handle(self, handle):
        return int.from_bytes(bytes(handle), "little")

    def ipc_close_handle(self, ptr):
        return None


def _desc(name, nbytes, src_ptr, dst_ptr, tag="weights_3"):
    return wx.XchgDesc(
        tag=tag, src_rank=CARD, dst_rank=CARD, param_name=name, kind=tp.FLAT,
        nbytes=nbytes, rows=1, run_bytes=nbytes, spitch=0, dpitch=0,
        src_ptr=src_ptr, dst_ptr=dst_ptr)


class _RingBase(unittest.TestCase):
    SIZES = [512 * 1024] * 8 + [int(2.5 * MIB)] + [300 * 1024] * 3
    SLOTS = 2

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="weg2-ring-h44-")
        self.nonce = f"h44r{os.getpid()}{int(time.time() * 1000) % 100000}"
        xr.unlink_semaphores(self.nonce)
        xr.create_semaphores(self.nonce)
        self._env_before = {k: os.environ.get(k) for k in (
            bx.SEQ_SYNC_BATCH_MIB_ENV, bx.SEQ_UNIT_DIGEST_ENV)}
        os.environ[bx.SEQ_SYNC_BATCH_MIB_ENV] = "1"
        os.environ[bx.SEQ_UNIT_DIGEST_ENV] = "1"
        self._ov = [envs.SGLANG_WEG2_SEQ_LANE_RING_SLOTS.override(self.SLOTS),
                    envs.SGLANG_WEG2_SEQ_LANE_RING_READY_MS.override(3000)]
        for o in self._ov:
            o.__enter__()
        self.dlines, self.clines = [], []

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

    # -- fixtures ------------------------------------------------------------
    def _plan(self, dops, cops, tag="weights_3", seed=0):
        descs, want = [], []
        for i, n in enumerate(self.SIZES):
            src, dst = 0x1000_0000 + i * 0x0100_0000, 0x4000_0000 + i * 0x0100_0000
            dops.hook(src, n)
            cops.hook(dst, n)
            data = hashlib.sha256(f"{tag}/{seed}/{i}".encode()).digest() * (n // 32 + 1)
            data = data[:n]
            dops.write(src, data)
            descs.append(_desc(f"u{i}", n, src, dst, tag=tag))
            want.append((dst, data))
        return descs, want

    def _run(self, descs, ops, phase, lines, **kw):
        return bx.run_sequential_units(
            descs, ops, self.nonce, slot_bytes=1 << 30, shm_root=self.root,
            phase=phase, card=CARD, log=lines.append, **kw)

    def _collect_thread(self, descs, cops, out, **kw):
        def body():
            out["rc"] = self._run(descs, cops, bx.PHASE_COLLECT, self.clines,
                                  dst_digest_fn=cops.digest, **kw)
        t = threading.Thread(target=body, daemon=True)
        t.start()
        return t

    def _ready_file(self, slot=0):
        lane_file = bx.seq_lane_file_name(f"c{CARD}", slot)
        return bx.seq_ready_path(bx.sequential_digest_path(self.nonce, self.root,
                                                           lane=lane_file))

    def _wait_ready(self, slot=0, timeout=10.0):
        p = self._ready_file(slot)
        t0 = time.time()
        while not os.path.exists(p):
            if time.time() - t0 > timeout:
                self.fail("the collector never announced readiness")
            time.sleep(0.002)

    def _round_trip(self, dops, cops, *, tag="weights_3", seed=0, buffer_slot=0):
        descs, want = self._plan(dops, cops, tag=tag, seed=seed)
        out = {}
        t = self._collect_thread(descs, cops, out, buffer_slot=buffer_slot)
        self._wait_ready(buffer_slot)
        rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines,
                       buffer_slot=buffer_slot)
        t.join(60)
        self.assertFalse(t.is_alive(), "collector hung")
        return rc, out.get("rc"), want

    def _assert_bytes(self, cops, want):
        for i, (dst, data) in enumerate(want):
            self.assertEqual(cops.read(dst, len(data)), data, f"unit {i} differs")

    def _lines(self, lines, marker):
        return [ln for ln in lines if marker in ln]

    def _n_batches(self):
        descs = [_desc(f"u{i}", n, 0, 0) for i, n in enumerate(self.SIZES)]
        return len(tp.batch_descs(descs, slot_bytes=MIB))


class RingRoundTrip(_RingBase):
    def test_bytes_bitgleich_with_wrap_and_a_unit_bigger_than_a_slot(self):
        dops, cops = _Ops(), _Ops()
        rc_d, rc_c, want = self._round_trip(dops, cops)
        self.assertEqual((rc_d, rc_c), ("", ""))
        self._assert_bytes(cops, want)
        n_b = self._n_batches()
        self.assertGreater(n_b, 2 * self.SLOTS, "the plan must wrap the ring")
        ring_d = self._lines(self.dlines, "WEG2-SEQ ring lane=c1 phase=deposit")
        ring_c = self._lines(self.clines, "WEG2-SEQ ring lane=c1 phase=collect")
        self.assertEqual(len(ring_d), 1, self.dlines)
        self.assertEqual(len(ring_c), 1, self.clines)
        for ln in ring_d + ring_c:
            self.assertIn(f"slots={self.SLOTS} slot_mib=1 tag_batches={n_b} "
                          f"wraps={(n_b - 1) // self.SLOTS}", ln)
            self.assertIn("wait_full_ms=", ln)
            self.assertIn("wait_free_ms=", ln)
        self.assertTrue(any("host=ring why=collector-ready" in ln
                            for ln in self.dlines), self.dlines)
        lt = self._lines(self.dlines + self.clines, "WEG2-SEQ lane-time lane=c1")
        self.assertEqual(len(lt), 2)
        self.assertTrue(all(ln.endswith("host=ring") for ln in lt), lt)
        # the ring file is the only lane file, sized hdr + R x slot
        d = os.path.dirname(bx.sequential_buffer_path(self.nonce, self.root, "c1"))
        names = sorted(os.listdir(d))
        self.assertIn("c1_ring.bin", names)
        self.assertNotIn("c1_unit_buffer.bin", names)
        self.assertEqual(os.path.getsize(os.path.join(d, "c1_ring.bin")),
                         bx.seq_ring_bytes(self.SLOTS, MIB))
        self.assertNotIn("c1_unit_digests.json.ready", names,
                         "the collector clears its readiness on the way out")
        # the host census booked the ring file
        cen = self._lines(self.dlines + self.clines, hl.LANES_MARKER)
        self.assertTrue(cen, "a new lane file must be a census event")
        self.assertIn("file=c1_ring.bin", cen[0])

    def test_the_same_ring_file_serves_the_next_tag(self):
        """Stale head from the previous tag (other token, higher freed) must
        not license the next tag's overwrite: the depositor resets it."""
        for k, tag in enumerate(("weights_3", "weights_5")):
            dops, cops = _Ops(), _Ops(copy_delay_s=0.002)
            rc_d, rc_c, want = self._round_trip(dops, cops, tag=tag, seed=k)
            self.assertEqual((rc_d, rc_c), ("", ""), (self.dlines[-3:], self.clines[-3:]))
            self._assert_bytes(cops, want)

    def test_collector_slower_than_depositor_the_depositor_waits_for_frees(self):
        dops, cops = _Ops(), _Ops(copy_delay_s=0.004)
        rc_d, rc_c, want = self._round_trip(dops, cops)
        self.assertEqual((rc_d, rc_c), ("", ""))
        self._assert_bytes(cops, want)
        ln = self._lines(self.dlines, "WEG2-SEQ ring lane=c1 phase=deposit")[0]
        waited = int(ln.split("waited_batches=")[1].split()[0])
        self.assertGreater(waited, 0, ln)

    def test_depositor_slower_than_collector_the_collector_waits_for_tokens(self):
        dops, cops = _Ops(copy_delay_s=0.004), _Ops()
        rc_d, rc_c, want = self._round_trip(dops, cops)
        self.assertEqual((rc_d, rc_c), ("", ""))
        self._assert_bytes(cops, want)
        ln = self._lines(self.clines, "WEG2-SEQ ring lane=c1 phase=collect")[0]
        wait_full = int(ln.split("wait_full_ms=")[1].split()[0])
        self.assertGreater(wait_full, 10, ln)

    def test_second_buffer_slot_has_its_own_ring_file(self):
        dops, cops = _Ops(), _Ops()
        rc_d, rc_c, want = self._round_trip(dops, cops, buffer_slot=1)
        self.assertEqual((rc_d, rc_c), ("", ""))
        self._assert_bytes(cops, want)
        d = os.path.dirname(bx.sequential_buffer_path(self.nonce, self.root, "c1"))
        self.assertIn("c1_s1_ring.bin", os.listdir(d))


def _child_deposit(root, nonce, sizes, slots, conn):
    """Der Depositor als eigener Prozess: eigene Ops, eigene Quelle."""
    try:
        os.environ[bx.SEQ_SYNC_BATCH_MIB_ENV] = "1"
        os.environ[bx.SEQ_UNIT_DIGEST_ENV] = "1"
        os.environ["SGLANG_WEG2_SEQ_LANE_RING_SLOTS"] = str(slots)
        os.environ["SGLANG_WEG2_SEQ_LANE_RING_READY_MS"] = "5000"
        ops = _Ops()
        descs = []
        for i, n in enumerate(sizes):
            src = 0x1000_0000 + i * 0x0100_0000
            ops.hook(src, n)
            data = hashlib.sha256(f"weights_3/0/{i}".encode()).digest() * (n // 32 + 1)
            ops.write(src, data[:n])
            descs.append(_desc(f"u{i}", n, src, 0x4000_0000 + i * 0x0100_0000))
        lines = []
        rc = bx.run_sequential_units(
            descs, ops, nonce, slot_bytes=1 << 30, shm_root=root,
            phase=bx.PHASE_DEPOSIT, card=CARD, log=lines.append)
        conn.send((rc, lines))
    except BaseException as exc:  # noqa: BLE001
        conn.send((f"child raised {type(exc).__name__}: {exc}", []))
    finally:
        conn.close()


class RingAcrossTwoProcesses(_RingBase):
    def test_two_processes_bitgleich(self):
        cops = _Ops(copy_delay_s=0.001)
        descs, want = [], []
        for i, n in enumerate(self.SIZES):
            dst = 0x4000_0000 + i * 0x0100_0000
            cops.hook(dst, n)
            data = hashlib.sha256(f"weights_3/0/{i}".encode()).digest() * (n // 32 + 1)
            want.append((dst, data[:n]))
            descs.append(_desc(f"u{i}", n, 0x1000_0000 + i * 0x0100_0000, dst))
        ctx = multiprocessing.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_child_deposit,
                        args=(self.root, self.nonce, self.SIZES, self.SLOTS, child))
        out = {}
        t = self._collect_thread(descs, cops, out)
        p.start()
        rc_d, dlines = parent.recv() if parent.poll(60) else ("child silent", [])
        p.join(30)
        t.join(60)
        self.assertEqual(rc_d, "", dlines[-5:])
        self.assertEqual(out.get("rc"), "", self.clines[-5:])
        self._assert_bytes(cops, want)
        self.assertTrue(any("WEG2-SEQ ring lane=c1 phase=deposit" in ln for ln in dlines))


class CollectorNotReadyKeepsOptionOne(_RingBase):
    def test_deposit_completes_without_its_collector_in_the_whole_form(self):
        with envs.SGLANG_WEG2_SEQ_LANE_RING_READY_MS.override(20):
            dops, cops = _Ops(), _Ops()
            descs, want = self._plan(dops, cops)
            t0 = time.time()
            rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines)
            self.assertEqual(rc, "")
            self.assertLess(time.time() - t0, 5.0)
            self.assertTrue(any("host=whole why=collector-not-ready" in ln
                                for ln in self.dlines), self.dlines)
            rc = self._run(descs, cops, bx.PHASE_COLLECT, self.clines,
                           dst_digest_fn=cops.digest)
            self.assertEqual(rc, "")
            self._assert_bytes(cops, want)
            lt = self._lines(self.dlines + self.clines, "WEG2-SEQ lane-time lane=c1")
            self.assertTrue(all(ln.endswith("host=whole") for ln in lt), lt)
            d = os.path.dirname(bx.sequential_buffer_path(self.nonce, self.root, "c1"))
            self.assertIn("c1_unit_buffer.bin", os.listdir(d))
            self.assertNotIn("c1_ring.bin", os.listdir(d))

    def test_mutant_ready_gate_always_true_stalls_the_deposit_W146(self):
        """MUTANT: without the ready gate the ring blocks a depositor whose
        collector cannot run yet -- exactly the #1374 cycle. It must end in
        the NAMED refusal, and only because the gate was removed."""
        with mock.patch.object(bx, "ring_collector_ready",
                               return_value=(True, "mutant")):
            dops, cops = _Ops(), _Ops()
            descs, _want = self._plan(dops, cops)
            t0 = time.time()
            rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines, budget_s=1.0)
        self.assertIn(bx.SEQ_RING_STALL_CODE, rc)
        self.assertIn("refusing instead of a silent stall", rc)
        self.assertLess(time.time() - t0, 20.0)


class ReadyButNobodyDrainsIsNamed(_RingBase):
    def test_stale_readiness_ends_in_W146(self):
        dops, cops = _Ops(), _Ops()
        descs, _want = self._plan(dops, cops)
        dpath = bx.sequential_digest_path(self.nonce, self.root, lane="c1")
        os.makedirs(os.path.dirname(dpath), exist_ok=True)
        bx.ring_post_ready(dpath, tag="weights_3", total_bytes=sum(self.SIZES))
        rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines, budget_s=1.0)
        self.assertTrue(rc.startswith(bx.SEQ_RING_STALL_CODE), rc)

    def test_readiness_of_another_tag_or_a_dead_pid_is_not_readiness(self):
        dpath = os.path.join(self.root, "x_unit_digests.json")
        bx.ring_post_ready(dpath, tag="weights_1", total_bytes=10)
        ok, why = bx.ring_collector_ready(dpath, tag="weights_2", total_bytes=10, wait_ms=0)
        self.assertFalse(ok)
        self.assertIn("ready-for-other-tag", why)
        ok, why = bx.ring_collector_ready(dpath, tag="weights_1", total_bytes=11, wait_ms=0)
        self.assertFalse(ok)
        import json
        with open(bx.seq_ready_path(dpath), "w") as fh:
            json.dump({"tag": "weights_1", "total": 10, "pid": 2 ** 22 + 12345}, fh)
        ok, why = bx.ring_collector_ready(dpath, tag="weights_1", total_bytes=10, wait_ms=0)
        self.assertFalse(ok)
        self.assertIn("ready-pid-gone", why)


class IpcTagsMapNoHostLane(_RingBase):
    def test_ipc_tag_creates_no_host_lane_file(self):
        """THE RAM CLAIM: 151/152 transfers of x148 rode IPC -- with the lazy
        form they map no host lane at all."""
        dops, cops = _IpcOps(), _IpcOps()
        # the collector opens the depositor's staging by its address: share
        # the depositor's hooks so the fake IPC address resolves
        descs, want = self._plan(dops, cops)
        cops.real.update(dops.real)
        cops.size.update(dops.size)
        cops.mem.update(dops.mem)
        rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines)
        self.assertEqual(rc, "")
        cops.real.update(dops.real)
        cops.size.update(dops.size)
        rc = self._run(descs, cops, bx.PHASE_COLLECT, self.clines,
                       dst_digest_fn=cops.digest)
        self.assertEqual(rc, "")
        self._assert_bytes(cops, want)
        d = os.path.dirname(bx.sequential_buffer_path(self.nonce, self.root, "c1"))
        lane_files = [n for n in os.listdir(d)
                      if n.endswith("unit_buffer.bin") or n.endswith("ring.bin")]
        self.assertEqual(lane_files, [], "an IPC tag must not map a host lane")
        lt = self._lines(self.dlines + self.clines, "WEG2-SEQ lane-time lane=c1")
        self.assertTrue(all("ipc=yes" in ln and ln.endswith("host=ipc") for ln in lt), lt)

    def test_switch_off_maps_the_host_lane_eagerly_even_for_ipc(self):
        with envs.SGLANG_WEG2_SEQ_LANE_RING.override(False):
            dops, cops = _IpcOps(), _IpcOps()
            descs, _want = self._plan(dops, cops)
            rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines)
            self.assertEqual(rc, "")
        d = os.path.dirname(bx.sequential_buffer_path(self.nonce, self.root, "c1"))
        self.assertIn("c1_unit_buffer.bin", os.listdir(d))


class SwitchOffIsTheOldPath(_RingBase):
    def test_switch_off_no_ready_no_ring_no_host_field(self):
        with envs.SGLANG_WEG2_SEQ_LANE_RING.override(False):
            dops, cops = _Ops(), _Ops()
            descs, want = self._plan(dops, cops)
            out = {}
            t = self._collect_thread(descs, cops, out)
            time.sleep(0.2)
            self.assertFalse(os.path.exists(self._ready_file()),
                             "the old collector announces nothing")
            rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines)
            t.join(60)
        self.assertEqual((rc, out.get("rc")), ("", ""))
        self._assert_bytes(cops, want)
        allk = self.dlines + self.clines
        self.assertFalse(any("WEG2-SEQ ring" in ln for ln in allk))
        self.assertFalse(any(" host=" in ln for ln in allk
                             if "WEG2-SEQ lane-time" in ln or "WEG2-SEQ mapped" in ln))
        d = os.path.dirname(bx.sequential_buffer_path(self.nonce, self.root, "c1"))
        self.assertNotIn("c1_ring.bin", os.listdir(d))
        # the old sync cadence: batch=32u/1MiB
        self.assertTrue(any("batch=32u/1MiB" in ln for ln in allk
                            if "lane-time" in ln))

    def test_cross_lanes_never_take_the_ring(self):
        """p* ride BAR1; their host fallback keeps its form untouched."""
        dops, cops = _Ops(), _Ops()
        descs, _want = self._plan(dops, cops)
        rc = bx.run_sequential_units(
            descs, dops, self.nonce, slot_bytes=1 << 30, shm_root=self.root,
            phase=bx.PHASE_DEPOSIT, pair=0, log=self.dlines.append)
        self.assertEqual(rc, "")
        self.assertFalse(any("ring-choice" in ln or " host=" in ln for ln in self.dlines))


class RingMutantsDie(_RingBase):
    def test_mutant_free_before_sync_corrupts_and_is_caught(self):
        """MUTANT: the collector frees a slot while its H2D copies are still
        queued (modelled: the collector's syncs do not flush) -- the
        depositor overwrites the slot and the late copies read the next
        batch's bytes. The placement check (dst digest) or the byte compare
        catches it; the honest order (flush in sync) passes above."""
        dops = _Ops()
        cops = _Ops(deferred=True)
        cops.flush_on_sync = False
        descs, want = self._plan(dops, cops)
        out = {}

        def body():
            out["rc"] = self._run(descs, cops, bx.PHASE_COLLECT, self.clines)
            cops.flush()

        t = threading.Thread(target=body, daemon=True)
        t.start()
        self._wait_ready()
        rc = self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines)
        t.join(60)
        self.assertEqual(rc, "")
        bad = [i for i, (dst, data) in enumerate(want) if cops.read(dst, len(data)) != data]
        self.assertTrue(bad, "free-before-sync must corrupt a destination")

    def test_mutant_depositor_ignores_the_free_is_caught(self):
        """MUTANT: the depositor reads the head as 'everything freed' -- it
        overruns a slow collector; the transport digest names the unit."""
        real_read = bx.ring_progress_read

        def lying(mm):
            tok, _freed = real_read(mm)
            return tok, 1 << 30

        dops, cops = _Ops(), _Ops(copy_delay_s=0.01)
        with mock.patch.object(bx, "ring_progress_read", side_effect=lying):
            rc_d, rc_c, want = self._round_trip(dops, cops)
        self.assertEqual(rc_d, "")
        corrupt = [i for i, (dst, data) in enumerate(want)
                   if cops.read(dst, len(data)) != data]
        self.assertTrue(("digest mismatch" in (rc_c or "")) or corrupt, rc_c)

    def test_mutant_record_without_token_check_is_refused(self):
        """A record from another ring token (a stale tag) is an identity
        mismatch, refused before the copy-out."""
        dops, cops = _Ops(), _Ops()
        descs, _want = self._plan(dops, cops)
        real_write = bx._ring_write_record

        def stale(dpath, g, rec):
            if g == 1:
                rec = dict(rec)
                rec["ring"] = dict(rec["ring"], token=int(rec["ring"]["token"]) ^ 0x5A5A)
            return real_write(dpath, g, rec)

        with mock.patch.object(bx, "_ring_write_record", side_effect=stale):
            out = {}
            t = self._collect_thread(descs, cops, out)
            self._wait_ready()
            self._run(descs, dops, bx.PHASE_DEPOSIT, self.dlines, budget_s=2.0)
            t.join(60)
        self.assertIn("unit identity mismatch at unit 1", out.get("rc") or "")


class LedgerPostenLanes(unittest.TestCase):
    def test_header_constant_is_one_number(self):
        self.assertEqual(hl.SEQ_RING_HDR_BYTES, bx.SEQ_RING_HDR_BYTES)

    def test_wake_slot_is_the_dead_diagonal_row(self):
        self.assertLess(bx.SEQ_RING_WAKE_SLOT, xr.SLOTS_PER_PAIR)
        self.assertNotEqual(bx.SEQ_RING_WAKE_SLOT, bx.CrossSlotRendezvous._COUNT_SLOT)
        self.assertNotEqual(bx.SEQ_RING_WAKE_SLOT, bx.CrossSlotRendezvous._DRAIN_SLOT)

    def test_priced_ring_bytes_for_the_rig(self):
        got = hl.seq_lanes_priced_bytes(ring_on=True, cards=3, depth=2, slots=4,
                                        slot_bytes=64 * MIB)
        self.assertEqual(got, 6 * (4096 + 256 * MIB))
        self.assertEqual(hl.seq_lanes_priced_bytes(ring_on=False, cards=3, depth=2,
                                                   slots=4, slot_bytes=64 * MIB), 0)

    def test_ledger_line_names_held_priced_and_over(self):
        ln = hl.lanes_ledger_line(event="new", lane_key="c1", file="c1_ring.bin",
                                  nbytes=4096 + 256 * MIB, files=2,
                                  ring_bytes=4096 + 256 * MIB,
                                  whole_bytes=562897920,
                                  priced_bytes=6 * (4096 + 256 * MIB))
        self.assertTrue(ln.startswith("WEG2-HOST-LEDGER LANES event=new lane=c1"))
        self.assertIn("priced=1.500 GiB", ln)
        self.assertIn("whole=0.524", ln)
        self.assertIn("over=0.000 GiB", ln)


if __name__ == "__main__":
    unittest.main()
