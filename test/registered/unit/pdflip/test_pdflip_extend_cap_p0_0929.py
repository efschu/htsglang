# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-EXTEND-CAP: under the P0 torch cache cap the D extend chunk follows the cap, not only the card.

METAL (p0-nopin, dkr27browauthorityp0nopinbar1fs09291720, 20cfcde893): pool 827584, D-TP1 (3080 nvml0)
cap 18609 MiB (PDFLIP-ALLOC-OVERHANG 17992 + 1470 - 853). 17:24:06Z a target extend of 4096 rows at
prefix 16434 died in barlink all_reduce: "Tried to allocate 40.00 MiB ... 133.75 MiB is free ... 18.17 GiB
allowed; 17.91 GiB allocated ... 251.86 MiB reserved but unallocated" -- the CAP refused (18593 + 40 >
18609), the card still had room. Before it: allocated 17354, reserved 18388 (round 17:24:02).
"""
import os
import unittest
from unittest import mock

try:
    from flliper.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from flliper.srt.pdflip import extend_trim as ET
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1 << 20
CAP = 18609
ALLOC = 17354          # live tensors before the extend
RESERVED = 18388       # reserved before the extend (cache 1034 above the live tensors)
UNRELEASABLE = 252     # what the allocator could not hand back at the line (reserved - allocated at the OOM)
CARD_FREE = 674        # card free before the extend (VRAM-PEAK round lines)
RATE = 0.3091          # D_EXTEND_GROWTH_PER_ROW_MIB (qwen27b)
DEATH_TRANSIENT = 18340 + 40 - ALLOC  # 1026 MiB at the refused allocation, layer 48 of 64


class _Cuda:
    def __init__(self):
        self.reserved = RESERVED
        self.free = CARD_FREE
        self.emptied = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return self.free * MIB, 20055 * MIB

    def memory_reserved(self, *a):
        return self.reserved * MIB

    def synchronize(self):
        pass

    def empty_cache(self):
        self.emptied += 1
        released = self.reserved - (ALLOC + UNRELEASABLE)
        self.reserved -= released
        self.free += released


def _env(cap=True):
    env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": f"{RATE},{RATE},{RATE}"}
    if cap:
        env.update({"FLLIPER_PDFLIP_TORCH_CACHE_CAP": "1", "FLLIPER_PDFLIP_TORCH_CACHE_CAP_MIB": f"29442,{CAP},18441"})
    return env


class ExtendCapP0Test(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def test_metal_extend_fits_the_cap(self):
        cuda = _Cuda()
        with mock.patch.dict(os.environ, _env(), clear=False):
            vote = ET.width_vote(cuda, 1, 4096, 1, True)
        self.assertIsNotNone(vote)                       # 4096 rows at the metal's state: cut
        self.assertLess(vote, 4096)
        self.assertEqual(cuda.emptied, 1)                # the cache above the live tensors goes first
        # the cut chunk's transient at the metal's per-row cost stays under the torch line
        self.assertLessEqual(ALLOC + UNRELEASABLE + vote * RATE, CAP)
        # and the metal's own 4096-row extend did not: 17354 + 252 + 1026 = 18632 > 18609
        self.assertGreater(ALLOC + UNRELEASABLE + DEATH_TRANSIENT, CAP)

    def test_rate_covers_the_death(self):
        self.assertGreaterEqual(RATE * 4096, DEATH_TRANSIENT)

    def test_no_cap_reads_the_card_alone(self):
        """Without the cap (every non-p0 profile) the vote is the card's, byte-identical."""
        cuda = _Cuda()
        cuda.free = 20000
        with mock.patch.dict(os.environ, _env(cap=False), clear=False):
            os.environ.pop("FLLIPER_PDFLIP_TORCH_CACHE_CAP", None)
            self.assertIsNone(ET.width_vote(cuda, 1, 4096, 1, True))
        self.assertEqual(cuda.emptied, 0)

    def test_cap_trim_once_per_extend(self):
        cuda = _Cuda()
        with mock.patch.dict(os.environ, _env(), clear=False):
            ET.width_vote(cuda, 1, 4096, 1, True)
            ET.width_vote(cuda, 1, 4096, 1, True)
            self.assertEqual(cuda.emptied, 1)

            class _Mode:
                def is_extend(self):
                    return True

                def is_target_verify(self):
                    return False

            batch = mock.Mock(forward_mode=_Mode())
            ET.before_extend(mock.Mock(is_draft_worker=False, tp_rank=1), batch)
            cuda.reserved = RESERVED
            ET.width_vote(cuda, 1, 4096, 1, True)
            self.assertEqual(cuda.emptied, 2)


class LauncherRateEnvTest(CustomTestCase):
    def test_record_rate_written_with_record_caps(self):
        from flliper.srt.pdflip import launcher as L

        ns = mock.Mock(profile="qwen27b", env_d="FLLIPER_PDFLIP_TORCH_CACHE_CAP=1")
        lines = []
        out = L._d_extend_cap_rate_env(ns, lines.append, "D")
        self.assertEqual(out, "0.3091,0.3091,0.3091")
        self.assertIn("FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB=0.3091,0.3091,0.3091", ns.env_d)
        self.assertTrue(any("EXTEND-CAP" in x for x in lines))


if __name__ == "__main__":
    unittest.main()
