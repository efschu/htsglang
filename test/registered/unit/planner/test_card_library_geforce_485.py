"""The card library must match the names NVML actually reports.

#485. `_canonical` exists to make lookup "case/space-insensitive" and strips the
vendor prefix -- but it strips only "nvidia", and every consumer NVIDIA card
reports as "NVIDIA GeForce RTX <n>". So:

    _canonical("NVIDIA GeForce RTX 5090") == "geforce rtx 5090"
    _canonical("RTX 5090")                == "rtx 5090"        <- the seed key

and `CardLibrary.has()` is False for every GeForce card on every rig, while
being True for the same card written the seed's way. `--pp-solve-cut` reads the
card name FROM THE CENSUS -- deliberately, so the torch-vs-NVML device-order
trap cannot mis-price a stage -- and the census records what the driver says.
So the flag refused with "no measured profile for 'NVIDIA GeForce RTX 5090'"
against a library that contains RTX 5090.

Caught by running the path against a real census rather than by reading it.
"""

import unittest

from sglang.srt.planner.card_library import CardLibrary, _canonical


class TheLibraryMatchesDriverReportedNamesTest(unittest.TestCase):
    def test_geforce_names_resolve_to_the_seed_entry(self):
        lib = CardLibrary()
        for driver_name, seed_name in (
            ("NVIDIA GeForce RTX 5090", "RTX 5090"),
            ("NVIDIA GeForce RTX 3080", "RTX 3080"),
            ("NVIDIA GeForce RTX 4090", "RTX 4090"),
        ):
            with self.subTest(driver_name):
                self.assertTrue(
                    lib.has(driver_name),
                    f"{driver_name!r} is what NVML reports and "
                    f"{seed_name!r} is in the library",
                )
                self.assertEqual(lib.get(driver_name).name, seed_name)

    def test_canonical_drops_both_vendor_words(self):
        self.assertEqual(_canonical("NVIDIA GeForce RTX 5090"), _canonical("RTX 5090"))
        self.assertEqual(_canonical("nvidia geforce rtx 3090"), _canonical("RTX 3090"))

    def test_it_does_not_collapse_distinct_models(self):
        """Loose matching must stay loose, not lossy."""
        self.assertNotEqual(_canonical("RTX 3080"), _canonical("RTX 3080 20GB"))
        self.assertNotEqual(_canonical("RTX 5090"), _canonical("RTX 5080"))


if __name__ == "__main__":
    unittest.main()
