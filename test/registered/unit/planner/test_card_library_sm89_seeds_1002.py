"""SM89-DURCHSPIEL-1002: the RTX 4090 / RTX 4080 catalogue entries stay
UNMEASURED seeds.

Both cards are in ``SEED_CARDS`` with nameplate capacity and peak specs
(sm89, fp8 660/390 TFLOPs) but NO measured GEMM/bandwidth -- ``--pp-solve-cut``
must keep refusing them ("carries no measured gemm/bandwidth rate") until the
rate pass ran on a real card. This pins the provenance so nobody promotes the
seeds to measured on the way: source == "seed", measured fields empty.
"""

import unittest

from flliper.srt.planner.card_library import seed_card

SM89_SEEDS = ("RTX 4090", "RTX 4080")


class TestSm89SeedsStayUnmeasured(unittest.TestCase):
    def test_seeds_are_present_sm89_and_unmeasured(self):
        for name in SM89_SEEDS:
            spec = seed_card(name)
            self.assertIsNotNone(spec, name)
            self.assertEqual(spec.sm_arch, "sm89", name)
            self.assertEqual(spec.source, "seed", name)
            self.assertIsNone(spec.gemm_tflops, name)
            self.assertIsNone(spec.membw_gbs, name)
            self.assertIsNone(spec.rate_env, name)

    def test_driver_reported_names_resolve_to_the_seed_capacities(self):
        # the NVML name carries the vendor words; the catalogue key does not.
        self.assertEqual(seed_card("NVIDIA GeForce RTX 4090").total_mib, 24564)
        self.assertEqual(seed_card("NVIDIA GeForce RTX 4080").total_mib, 16376)


if __name__ == "__main__":
    unittest.main()
