"""y7h (23c8fb584e): on ranks that see ONE device, a CUDA query for another card's
launcher ordinal fails AND leaves cudaErrorInvalidDevice as the thread's last
error; the next checked launch raises it far away (D's resume: "invalid device
ordinal"). Bar1Lanes.big_cards asked bdf_of_card(c) for every card of the pairs
(y7h: `big_cards: card 1 unreadable ... unknown-1` on TP1/TP2/PP1/PP2).

Now: own card -> the rank's own BDF (the one ordinal it may ask CUDA about);
other cards -> the source's mapped peer window (PeerWindow.peer_bdf) or the
per-boot card registry each process publishes its own BDF into. And a failed
cudaDeviceGetPCIBusId reads its error away (cudaGetLastError).
"""

import inspect
import tempfile
import types
import unittest

from sglang.srt.distributed.device_communicators import barlink_bar1, barlink_matrix
from sglang.srt.weg2 import bar1_lanes as bl

PAIRS = [(1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2)]
BDF = {0: "0000:01:00.0", 1: "0000:02:00.0", 2: "0000:03:00.0"}
SIZE = {"0000:01:00.0": 32 << 30, "0000:02:00.0": 256 << 20, "0000:03:00.0": 256 << 20}


class BigCardsOnAOneDeviceRank(unittest.TestCase):
    def setUp(self):
        self.asked = []

        def _bdf_of_card(device):
            d = device.index if hasattr(device, "index") else int(device)
            self.asked.append(d)
            if d != 0:  # this process sees ONE device: ordinal 0 is its own card
                raise AssertionError("invalid device ordinal")
            return self.own

        self._orig = (barlink_matrix.bdf_of_card, barlink_bar1.bar1_window)
        barlink_matrix.bdf_of_card = _bdf_of_card
        barlink_bar1.bar1_window = lambda bdf: types.SimpleNamespace(size=SIZE[bdf])
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        barlink_matrix.bdf_of_card, barlink_bar1.bar1_window = self._orig
        self.tmp.cleanup()

    def _lanes(self, rank, group="D"):
        self.own = BDF[rank]
        return bl.Bar1Lanes("n1", group, rank, 0, PAIRS, log=lambda *_: None, root=self.tmp.name)

    def test_no_cuda_query_for_a_foreign_ordinal(self):
        # card 0 (the big one) published itself, as its own process does at setup
        r0 = self._lanes(0)
        r0.bdf()
        r1 = self._lanes(1)
        big = r1.big_cards()
        self.assertEqual(big, [0])                       # named via the registry
        self.assertEqual([a for a in self.asked if a != 0], [])
        self.assertEqual(self.asked.count(0), 2)         # each rank asked its OWN device once

    def test_peer_window_names_the_receiver(self):
        r1 = self._lanes(1)
        k = PAIRS.index((1, 0))
        r1.peers[f"p{k}"] = types.SimpleNamespace(peer_bdf=BDF[0])
        self.assertEqual(r1.card_bdf(0), BDF[0])
        self.assertEqual([a for a in self.asked if a != 0], [])

    def test_an_unnamed_card_is_not_big_and_not_asked(self):
        r2 = self._lanes(2)
        self.assertEqual(r2.big_cards(), [])
        self.assertEqual([a for a in self.asked if a != 0], [])


class AFailedQueryClearsItsError(unittest.TestCase):
    def test_bdf_of_card_reads_the_error_away(self):
        src = inspect.getsource(barlink_matrix.bdf_of_card)
        self.assertIn("cudaGetLastError", src)
        self.assertLess(src.index("cudaDeviceGetPCIBusId"), src.index("cudaGetLastError"))


if __name__ == "__main__":
    unittest.main()
