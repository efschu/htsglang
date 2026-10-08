# SPDX-License-Identifier: Apache-2.0
"""D-LEAD-MS-1007: the scheduler arrival and the first extend of a request on
group D, in epoch milliseconds, one line per rank.

The D log stamps to the second; the 0.42-0.55 s scheduler share of the D lead
(Voranalyse 07.10.) could not be read off it. The line is the measurement the
metal joins with the front's D-ADMIT / LEG2-FIRST-CONTENT by rid. It is log
only: off by default, and nowhere but group D.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_lead_probe as L  # noqa: E402

LINE = re.compile(r"PDFLIP D-LEAD-MS rid=(\S+) recv_ms=(\d+) extend_ms=(\d+) "
                  r"recv_to_extend_ms=([\d.]+) skip=([01])$")


class Probe(unittest.TestCase):
    def test_built_only_on_group_d_with_the_switch(self):
        self.assertFalse(envs.FLLIPER_LOG_PDFLIP_D_LEAD_MS.get())
        self.assertIsNone(L.build_d_lead_probe(group="D"))
        with envs.FLLIPER_LOG_PDFLIP_D_LEAD_MS.override(True):
            self.assertIsNotNone(L.build_d_lead_probe(group=" d"))
            for group in ("P", "", "DUAL"):
                self.assertIsNone(L.build_d_lead_probe(group=group))

    def test_one_line_at_the_first_extend_with_both_stamps(self):
        probe = L.DLeadProbe(clock=iter([100.0, 100.4375, 101.0]).__next__)
        with self.assertLogs(L.logger, logging.INFO) as logs:
            probe.note_recv(rid="pdflip-3-9")
            probe.note_extend(rids=["pdflip-3-9", "pdflip-x"], skip=False)
            probe.note_extend(rids=["pdflip-3-9"], skip=True)  # a later chunk: no line
            L.logger.info("end")
        lines = [r.getMessage() for r in logs.records if "D-LEAD-MS" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        m = LINE.search(lines[0])
        self.assertIsNotNone(m, lines[0])
        self.assertEqual(m.groups(), ("pdflip-3-9", "100000", "100437", "437.5", "0"))

    def test_pending_rids_are_bounded(self):
        probe = L.DLeadProbe()
        for i in range(L.PENDING_CAP + 3):
            probe.note_recv(rid=f"r{i}")
        with self.assertLogs(L.logger, logging.INFO) as logs:
            probe.note_extend(rids=["r0", "r1", "r2", "r3"], skip=False)
            L.logger.info("end")
        self.assertEqual([r.getMessage().split()[2] for r in logs.records
                          if "D-LEAD-MS" in r.getMessage()], ["rid=r3"])


class Wiring(unittest.TestCase):
    def test_scheduler_notes_the_arrival_and_the_extend(self):
        from flliper.srt.managers.scheduler import Scheduler

        self.assertIsNone(Scheduler.pdflip_d_lead_probe)
        self.assertIn("self.pdflip_d_lead_probe.note_recv(rid=recv_req.rid)",
                      inspect.getsource(Scheduler.handle_generate_request))
        src = inspect.getsource(Scheduler._run_batch_forward)
        self.assertIn("self.pdflip_d_lead_probe is not None and batch.forward_mode.is_extend()", src)
        self.assertIn('group=os.environ.get("FLLIPER_PDFLIP_GROUP", "")',
                      inspect.getsource(Scheduler.init_pdflip_d_lead_probe))


if __name__ == "__main__":
    unittest.main()
