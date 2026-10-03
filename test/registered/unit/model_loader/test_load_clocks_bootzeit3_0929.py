# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 3 (29.09.): the loader thread's time, split (instrument only).

Metal (rc12z30o3, NF P+D): every rank's LOAD-PROFILE showed ``threading.py
wait`` at 49-59 % of the loader thread and the stream window full
(inflight_peak = budget), but no line said WHAT the thread waited for. The
clocks below split it; the next boot names the lever:

* ``ExpertLoadPool``: wait_slots_s (blocked on a consumer slot), deferred_s
  (running the per-layer presplit), drain_wait_s (waiting for the tail);
* ``WEG2-LOAD-COALESCE``: wait_read_s (a read had not landed) versus
  consume_s (the consumer held the tensor).
"""

import logging
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.model_loader import load_consumer as lc
from sglang.srt.model_loader import weight_utils as W


class TestPoolClocks(unittest.TestCase):
    def test_slot_wait_is_counted_when_consumers_are_behind(self):
        pool = lc.ExpertLoadPool(1, in_flight=1)
        gate = threading.Event()
        pool.submit(gate.wait)
        threading.Timer(0.15, gate.set).start()
        pool.submit(lambda: None)  # blocks on the only slot until the gate opens
        pool.drain()
        pool.close()
        self.assertGreaterEqual(pool.wait_slots_s, 0.1)
        self.assertEqual(pool.completed, 2)

    def test_deferred_time_is_counted_on_the_loader_thread(self):
        pool = lc.ExpertLoadPool(2)
        pool.submit(lambda: pool.defer_to_loader(lambda: time.sleep(0.05)))
        pool.drain()
        pool.close()
        self.assertEqual(pool.deferred_run, 1)
        self.assertGreaterEqual(pool.deferred_s, 0.04)
        self.assertGreaterEqual(pool.drain_wait_s, 0.0)

    def test_serial_form_counts_deferred_inline_and_never_waits(self):
        pool = lc.ExpertLoadPool(0)
        pool.submit(lambda: pool.defer_to_loader(lambda: time.sleep(0.02)))
        pool.drain()
        self.assertEqual(pool.deferred_run, 1)
        self.assertGreaterEqual(pool.deferred_s, 0.015)
        self.assertEqual(pool.wait_slots_s, 0.0)
        self.assertEqual(pool.drain_wait_s, 0.0)

    def test_drain_wait_counts_the_tail(self):
        pool = lc.ExpertLoadPool(1)
        pool.submit(time.sleep, 0.1)
        pool.drain()
        pool.close()
        self.assertGreaterEqual(pool.drain_wait_s, 0.05)


class TestCoalesceLineSplitsReadFromConsume(unittest.TestCase):
    def test_line_carries_wait_read_and_consume(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model-00001-of-00001.safetensors")
            save_file({f"w{i}": torch.full((64, 64), float(i)) for i in range(8)}, path)
            with mock.patch.dict(os.environ, {W.COALESCE_ENV: "1"}), \
                    self.assertLogs(W.logger, logging.INFO) as cm:
                n = 0
                for _name, _t in W.pread_safetensors_stream(
                        [path], None, direct_io=False, workers=2, log=True):
                    time.sleep(0.01)  # the consumer holds each tensor
                    n += 1
            self.assertEqual(n, 8)
            line = next(r for r in cm.output if W.COALESCE_MARKER in r)
            self.assertIn("wait_read_s=", line)
            consume = float(line.split("consume_s=")[1].split()[0])
            self.assertGreaterEqual(consume, 0.07)


if __name__ == "__main__":
    unittest.main()
