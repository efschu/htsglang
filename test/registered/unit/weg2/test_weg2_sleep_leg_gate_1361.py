# SPDX-License-Identifier: Apache-2.0
"""#1361 fix6 -- the gate at the START of the sleep leg.

W98 fired 4 seconds before boot weg2xsn25's first OOM kill and prevented
nothing. The reason is structural, not a tuning failure: the STOP stops
ADMISSION, and the bytes that spent the cushion were D's own sleep leg, already
3.3 s into a 5.6 s RPC when the latch fired.

Measured on that boot:

    07:00:50..07:01:09  cushion FLAT at 2.697 GiB, PSI 0.00
    ~07:01:07.6         the sleep leg starts
    07:01:10.0 -> .5    cushion 2.591 -> 0.548 in ONE 0.5 s tick
    07:01:10.870        W98 latches, WEG2 STOP
    07:01:13.188        the sleep RPC returns (ms=5572)
    07:01:14.6..24.8    first rank killed, global host OOM

Widening the latch floor does not help, and that was measured rather than
assumed: floors 1.0, 1.5, 2.0, 2.5, 2.60 and 2.69 GiB ALL latch on the SAME
sample with the SAME 4.1 s lead, because the cushion is flat for a minute and
then vertical inside one tick. Only 3.0 moves it one tick; 5.0 buys 46 s while
firing 8 times on the SURVIVING boot weg2xsn24.

So the lever is here, where the arithmetic is still ordinary: 2.697 GiB of
cushion against a 4.32 GiB write, knowable ~3 s before the first byte.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

#: weg2xsn25, the boot that died. Cushion from the last sample before the
#: write; need from that same boot's own recorded `shmem_delta_gib`.
XSN25_CUSHION = 2.697
XSN25_NEED = 4.32
#: weg2xsn26, the boot that came through its sleep leg. Peak non-reclaimable
#: 84.28 GiB over n=964, riegel never fired, oom_kill 5 -> 5.
XSN26_CUSHION = 8.11


class TheLegIsRefusedBeforeTheFirstByte(CustomTestCase):
    def test_red_on_the_xsn25_numbers(self):
        with self.assertRaises(hl.Weg2SleepLegCushionDeficit) as cm:
            hl.refuse_sleep_leg_deficit(XSN25_CUSHION, XSN25_NEED,
                                        source="record:weg2xsn25", group="D")
        m = str(cm.exception)
        for token in ("W100", "cushion_gib=2.70", "need_gib=4.32",
                      "margin=0.00", "source=record:weg2xsn25"):
            self.assertIn(token, m, f"the refusal does not name {token}")

    def test_green_on_the_xsn26_numbers(self):
        """The boot that survived its sleep leg must not be refused.

        This is the negative control, and it is the half that decides whether
        the gate is a gate or a boot killer.
        """
        self.assertIsNone(
            hl.refuse_sleep_leg_deficit(XSN26_CUSHION, XSN25_NEED,
                                        source="record:weg2xsn26", group="D"))

    def test_neither_term_is_guessed_when_it_is_missing(self):
        """`None` is not zero on either side.

        An unreadable cushion is not "no cushion left"; an unknown write size
        is not "writes nothing". Both would turn a blind gate into a boot
        killer at the one moment where a wrong refusal costs the whole window.
        """
        self.assertIsNone(hl.refuse_sleep_leg_deficit(None, XSN25_NEED))
        self.assertIsNone(hl.refuse_sleep_leg_deficit(XSN25_CUSHION, None))

    def test_it_is_a_ledger_refusal_so_the_teardown_funnel_catches_it(self):
        self.assertTrue(
            issubclass(hl.Weg2SleepLegCushionDeficit, hl.Weg2HostLedgerRefused))


class TheCallSiteIsWiredNotJustThePureFunction(CustomTestCase):
    """THE EXECUTION SMOKE. A pure function nobody calls is the state this seat
    has paid for four times this week -- and the xsn26 class specifically: the
    D path did not pass `terms` through and the leg died on a refusal whose
    unit tests were all green."""

    def _front(self, tmpdir, delta):
        import json

        from sglang.srt.weg2 import front as fr

        path = os.path.join(tmpdir, "rec.json")
        with open(path, "w") as f:
            json.dump({"samples": [{
                "group": "D", "boot_tag": "weg2xsn25", "at": "2026-09-13T07:00Z",
                "shmem_delta_gib": delta, "rss_shmem_gib": 48.0,
            }]}, f)
        f_ = fr.Front.__new__(fr.Front)
        f_.measured_record = path
        return f_

    def test_the_gate_runs_from_the_real_flip_path(self):
        """The line is IN `_drain_and_flip`'s gathered-leg block, not beside it."""
        import inspect

        from sglang.srt.weg2 import front as fr

        src = inspect.getsource(fr.Front)
        gather = src.index('self.timed_rpc(S, "/release_memory_occupation"')
        call = src.index("self._sleep_leg_gate(")
        self.assertLess(call, gather,
                        "the gate must run BEFORE the leg is issued")
        self.assertLess(gather - call, 1200,
                        "the gate drifted away from the leg it guards")

    def test_executing_the_call_site_refuses_on_the_xsn25_numbers(self):
        """Not a source assertion: the method is CALLED and must raise."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            f_ = self._front(td, XSN25_NEED)
            with mock.patch.object(hl, "read_cgroup_pressure",
                                   return_value={"file_gib": 60.0 + XSN25_CUSHION,
                                                 "shmem_gib": 60.0}):
                with self.assertRaises(hl.Weg2SleepLegCushionDeficit):
                    f_._sleep_leg_gate("D")

    def test_executing_the_call_site_passes_on_the_xsn26_numbers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            f_ = self._front(td, XSN25_NEED)
            with mock.patch.object(hl, "read_cgroup_pressure",
                                   return_value={"file_gib": 60.0 + XSN26_CUSHION,
                                                 "shmem_gib": 60.0}):
                self.assertIsNone(f_._sleep_leg_gate("D"))

    def test_a_front_without_a_sidecar_proceeds(self):
        """Every pre-fix6 deployment path stays byte-identical."""
        from sglang.srt.weg2 import front as fr

        f_ = fr.Front.__new__(fr.Front)
        f_.measured_record = ""
        self.assertIsNone(f_._sleep_leg_gate("D"))


if __name__ == "__main__":
    unittest.main()
