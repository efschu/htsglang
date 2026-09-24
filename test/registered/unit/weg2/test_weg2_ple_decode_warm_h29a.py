"""fnFL2 H29a (SGLANG_WEG2_PLE_DECODE_PREFETCH): D's PLE mapping warm from P's rows.

The decode PLE gather is an in-graph HMM kernel on TP0; D never prefilled the
prompt, so its own mapping of the checkpoint shards is cold for the prompt's
n-gram rows. P publishes its last prefill gather's rows; D faults their pages
into its mapping on a background thread, never touching an address mincore
does not see mapped.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging
import mmap
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.models import qwen4_exp_ple_table as pt
from sglang.srt.weg2 import decode_warm_handoff as dwh
from sglang.test.test_utils import CustomTestCase

class TestPleRows(CustomTestCase):
    def test_select_repeated_rows_first_then_the_latest_tail_rows(self):
        ids = np.array([7, 3, 9, 3, 7, 3, 11, 12, 13])
        out = dwh.select_ple_rows(ids, max_rows=10, tail_rows=4)
        self.assertEqual(out.tolist(), [3, 7, 13, 12, 11])
        self.assertEqual(dwh.select_ple_rows(ids, max_rows=2, tail_rows=4).tolist(), [3, 7])

    def test_publish_and_load_round_trip_and_age(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(dwh.publish_ple_rows(np.array([5, 5, 6]), d))
            rows, mtime = dwh.load_ple_rows(d, 600)
            self.assertEqual(rows.tolist(), [5, 6])
            self.assertIsNotNone(mtime)
            path = os.path.join(d, dwh.PLE_ROWS_FILE)
            os.utime(path, (time.time() - 5000, time.time() - 5000))
            self.assertEqual(dwh.load_ple_rows(d, 600), (None, None))

    def test_publish_armed_by_switch_and_never_on_d(self):
        with envs.SGLANG_WEG2_PLE_DECODE_PREFETCH.override(False):
            self.assertFalse(pt.note_ple_prefill_rows(torch.tensor([1, 2])))
        with envs.SGLANG_WEG2_PLE_DECODE_PREFETCH.override(True), \
                mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            self.assertFalse(pt.note_ple_prefill_rows(torch.tensor([1, 2])))


def _mapped_table(tmpdir, rows_per_shard=64, shards=2, dim=160):
    """A real read-only mapping of a file laid out like the checkpoint table."""
    row_bytes = dim * 2
    path = os.path.join(tmpdir, "ple.bin")
    with open(path, "wb") as f:
        f.write(os.urandom(rows_per_shard * shards * row_bytes))
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    os.close(fd)
    base = np.frombuffer(mm, dtype=np.uint8).ctypes.data
    bases = [base + s * rows_per_shard * row_bytes for s in range(shards)]
    table = pt.CheckpointMappedPleTable(
        bases, rows_per_shard, rows_per_shard * shards, torch.bfloat16, dim,
        keepalive=[mm], files=[path],
    )
    return table, mm


class TestPleDecodeWarm(CustomTestCase):
    def test_pages_in_row_order_straddling_rows_name_both_pages(self):
        with tempfile.TemporaryDirectory() as d:
            table, mm = _mapped_table(d)
            # row 12 = bytes 3840..4159 straddles page 0 / page 1
            pages = pt.ple_row_pages(table, [12, 0, 12, 10**9, -1])
            b0 = table.bases[0] >> 12 << 12
            self.assertEqual(pages, [b0, b0 + 4096])
            del table
            mm.close()

    def test_census_touches_only_mapped_pages(self):
        with tempfile.TemporaryDirectory() as d:
            table, mm = _mapped_table(d)
            c = pt.warm_ple_rows(table, list(range(0, 128, 5)), max_pages=0)
            self.assertEqual(c["warm"] + c["cold"] + c["unmapped"], c["pages"])
            self.assertEqual(c["unmapped"], 0)
            self.assertEqual(c["touched"], c["cold"])
            again = pt.warm_ple_rows(table, list(range(0, 128, 5)), max_pages=1)
            self.assertEqual(again["pages"], 1)
            # an address outside every mapping is counted, never dereferenced
            bogus = pt.CheckpointMappedPleTable(
                [4096], 64, 64, torch.bfloat16, 160, keepalive=[], files=[]
            )
            c = pt.warm_ple_rows(bogus, [0, 1], max_pages=0)
            self.assertEqual((c["unmapped"], c["touched"]), (1, 0))
            del table
            mm.close()

    def test_start_warms_once_per_published_file_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            table, mm = _mapped_table(d)
            model = torch.nn.Module()
            emb = torch.nn.Module()
            emb._ckpt_table = table
            model.add_module("emb", emb)
            dwh.publish_ple_rows(np.array([1, 1, 70, 3]), d)
            pt._PLE_WARM_LAST_MTIME = None
            with self.assertLogs("sglang.srt.models.qwen4_exp_ple_table", "INFO") as cm:
                pt.start_ple_decode_warm(model, directory=d, background=False)
            line = [x for x in cm.output if "PLE-DECODE-PREFETCH" in x][0]
            self.assertIn("rows=3 pages=2", line)
            # the same file is not warmed twice
            with mock.patch.object(pt, "warm_ple_rows") as w:
                pt.start_ple_decode_warm(model, directory=d, background=False)
                w.assert_not_called()
            del emb._ckpt_table, table
            mm.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
