"""y7h (23c8fb584e) died at its first P->D wake: D TP1/TP2
`resume_memory_occupation FAILED: AcceleratorError: CUDA error: invalid device
ordinal`. FLIPCYCLE H4's gate named each receiver card by its launcher ordinal
through barlink_matrix.bdf_of_card -> cudaDeviceGetPCIBusId on ranks that see
ONE device; the failure was swallowed, the runtime kept the error as the thread's
last error, and D's next checked launch (the resume) raised it.

The gate now reads the receiver's PCI address from the lane's own handshake
(PeerWindow.peer_bdf) and makes no CUDA call at all.
"""

import types
import unittest

from flliper.srt.distributed.device_communicators import barlink_matrix
from flliper.srt.pdflip import lane_priority as lp


class _Lanes:
    def __init__(self, peers):
        self.cross_pairs = [(0, 1), (0, 2)]
        self.peers = peers

    def role(self, lk):
        return "src"


class NoCudaOrdinalLookup(unittest.TestCase):
    def setUp(self):
        self.called = []

        def _boom(device):
            self.called.append(device)
            raise RuntimeError("cudaDeviceGetPCIBusId: invalid device ordinal")

        self._orig_bdf = barlink_matrix.bdf_of_card
        barlink_matrix.bdf_of_card = _boom
        self._orig_rate = lp.link_gbytes_per_s
        rates = {"0000:01:00.0": 7.9, "0000:02:00.0": 15.8}
        lp.link_gbytes_per_s = lambda bdf, root=None: rates.get(bdf)

    def tearDown(self):
        barlink_matrix.bdf_of_card = self._orig_bdf
        lp.link_gbytes_per_s = self._orig_rate

    def test_the_gate_never_asks_cuda_for_a_foreign_ordinal(self):
        lanes = _Lanes({"p0": types.SimpleNamespace(peer_bdf="0000:01:00.0"),
                        "p1": types.SimpleNamespace(peer_bdf="0000:02:00.0")})
        g = lp.gate_for(lanes)
        self.assertEqual(self.called, [], "bdf_of_card (a CUDA call) must not run")
        self.assertIsNotNone(g)
        self.assertEqual(g.top, "p1")

    def test_a_lane_without_a_mapped_peer_has_no_rate(self):
        lanes = _Lanes({"p0": types.SimpleNamespace(peer_bdf="0000:01:00.0")})
        self.assertIsNone(lp.gate_for(lanes))  # one rate only: no order to keep
        self.assertEqual(self.called, [])


if __name__ == "__main__":
    unittest.main()
