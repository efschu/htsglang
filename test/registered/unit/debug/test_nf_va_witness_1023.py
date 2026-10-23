"""NF #162 (23.10.): the rank's device VA map on a CUDA-error death (debug_utils/va_witness.py).

rc31b D-TP1 faulted at VA 0x5_4a4e0000 (FAULT_PDE) and no log line named the base VA of any tensor, so the address had
no owner. The map written at death names, for every page-controller tensor, its base, size and the spans mapped at
that moment, plus the torch segments; ``va_lookup`` answers "whose is this address, and was it mapped".
"""
import ast
import json
import logging
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PY = os.path.join(ROOT, "python")
sys.path.insert(0, PY)

from flliper.srt.debug_utils import va_witness as vw  # noqa: E402

MiB = 1 << 20


@dataclass
class _RowGeom:
    name: str
    rows_boot: int
    rows_max: int
    row_bytes: int
    alloc_bytes: int


@dataclass
class _M:
    ptr: int
    geom: object


class _Ctl:
    def __init__(self):
        # a bank tensor of 24 rows x 2 MiB at 0x5_4a000000; rows 0..12 mapped (cap form + 1 lent row), the rest OFF
        self.row_tensors = [_M(0x5_4A000000, _RowGeom("35.w13_weight", 12, 24, 2 * MiB, 48 * MiB))]
        self.slot_tensors = []
        self.kv_tensors = []
        self.spans_by_ptr = {0x5_4A000000: ((0, 13 * 2 * MiB),)}


class _Sched:
    pass


def _sched():
    s = _Sched()
    setattr(s, vw.CTL_ATTR, _Ctl())
    return s


_SEGS = [{"address": 0x3_02000000, "total_size": 64 * MiB, "allocated_size": 32 * MiB, "stream": 0,
          "segment_pool_id": (0, 0), "device": 0}]


class TheMapNamesTheOwnerOfAFaultAddress(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, sched):
        with mock.patch.object(vw, "torch_segments", lambda: [
                {"base": s["address"], "size": s["total_size"], "allocated": s["allocated_size"],
                 "stream": s["stream"], "pool": str(s["segment_pool_id"]), "device": 0} for s in _SEGS]):
            with self.assertLogs("t_va", level="ERROR") as cm:
                path = vw.write_va_map(sched, "D TP1 PP0", "cuda_error", dump_dir=self.tmp.name,
                                       log=logging.getLogger("t_va"))
        return path, cm.output

    def test_one_file_on_the_dump_directory_and_one_log_line(self):
        path, out = self._write(_sched())
        self.assertTrue(path.startswith(self.tmp.name + "/va_map_D_TP1_PP0_"))
        self.assertEqual(len(out), 1)
        self.assertIn("VA-MAP rank=D TP1 PP0", out[0])
        self.assertIn("managed=1 [0x54a000000,0x54d000000)", out[0])
        self.assertIn("segments=1 [0x302000000,0x306000000)", out[0])

    def test_the_rc31b_address_resolves_to_an_unmapped_bank_row(self):
        path, _ = self._write(_sched())
        m = json.load(open(path))
        hits = vw.va_lookup(m, 0x5_4A4E0000)  # offset 0x4e0000 = row 2 -> mapped
        self.assertEqual(hits, ["managed row_tensors 35.w13_weight offset=0x4e0000 row=2 mapped"])
        hits = vw.va_lookup(m, 0x5_4A000000 + 20 * 2 * MiB + 0x10000)  # row 20 -> OFF
        self.assertEqual(hits, ["managed row_tensors 35.w13_weight offset=0x2810000 row=20 UNMAPPED"])
        self.assertEqual(vw.va_lookup(m, 0x3_02100000)[0].split(" base=")[0], "torch segment")
        self.assertEqual(vw.va_lookup(m, 0x7_00000000), [])  # nobody: freed segment or foreign VA

    def test_no_controller_still_writes_the_segments(self):
        path, out = self._write(_Sched())
        m = json.load(open(path))
        self.assertEqual(m["managed"], [])
        self.assertEqual(len(m["segments"]), 1)

    def test_a_hanging_collector_is_bounded_and_named(self):
        import threading
        ev = threading.Event()
        with mock.patch.object(vw, "_collect", lambda sched, out: ev.wait(10)):
            with self.assertLogs("t_va", level="ERROR") as cm:
                path = vw.write_va_map(_sched(), "D TP1 PP0", "cuda_error", dump_dir=self.tmp.name,
                                       log=logging.getLogger("t_va"), timeout_s=0.2)
        ev.set()
        self.assertIn("TIMEOUT", cm.output[0])
        self.assertEqual(json.load(open(path))["timeout_s"], 0.2)

    def test_it_never_raises(self):
        with self.assertLogs("t_va", level="ERROR") as cm:
            res = vw.write_va_map(_sched(), "D TP1 PP0", "cuda_error", dump_dir="/proc/definitely/not",
                                  log=logging.getLogger("t_va"))
        self.assertIsNone(res)
        self.assertIn("not written", cm.output[0])

    def test_switched_off_without_a_dump_dir_it_writes_nothing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLLIPER_CUDA_COREDUMP", None)
            self.assertIsNone(vw.write_va_map(_sched(), "D TP1 PP0", "cuda_error"))


class TheCtlAttrMatchesTheController(unittest.TestCase):
    def test_same_attribute_name_as_d_seat_vram(self):
        src = open(os.path.join(PY, "flliper", "srt", "pdflip", "d_seat_vram.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        val = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "CTL_ATTR" for t in n.targets))
        self.assertEqual(val, vw.CTL_ATTR)


class TheDeathPathWritesItOnACudaError(unittest.TestCase):
    def test_the_scheduler_except_block_calls_write_va_map_after_the_traceback_line(self):
        src = open(os.path.join(PY, "flliper", "srt", "managers", "scheduler.py"), encoding="utf-8").read()
        i_tb = src.index('logger.error(f"Scheduler hit an exception: {traceback}")')
        i_va = src.index('_va_witness.write_va_map(scheduler, _cudacore_label, "cuda_error", log=logger)')
        i_quit = src.index("_pdflip_rankstats.note_stop(scheduler_exc)")
        self.assertLess(i_tb, i_va)
        self.assertLess(i_va, i_quit)


if __name__ == "__main__":
    unittest.main()
