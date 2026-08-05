"""The #583 soak harness greps #581's log markers -- pin them to the sources.

The soak window (#583 barlink abort path + #581 mamba pool floor) reads its
acceptance out of the server log by matching fixed strings. That is only sound
while those strings exist. A reworded log message would silently turn every
acceptance into "not observed", and "not observed" is exactly what a passing
soak looks like for the negative half -- so the failure would be invisible in
the one direction that matters.

This test fails instead, naming the marker and the file it drifted out of.
"""

import re
import unittest
from pathlib import Path

_HARNESS = Path("scripts/repro/barlink_launch_failure_583.sh")
_SRT = Path("python/sglang/srt")

#: literal -> source file that must contain it verbatim.
#:
#: NOTE on "deferring this batch": the format string in memory_pool.py wraps
#: between "batch " and "of %d request(s)", so only this shorter fragment is
#: contiguous in the SOURCE. The harness greps the same fragment for exactly
#: that reason -- a longer pin would match the formatted log line but could
#: not be verified here, which is how a pin rots unnoticed.
PINNED_MARKERS = {
    "mamba write-through pin budget reached": "mem_cache/hi_mamba_radix_cache.py",
    "deferring this batch": "mem_cache/memory_pool.py",
    "skipping this cache insert": (
        "mem_cache/unified_cache_components/mamba_component.py"
    ),
    "mamba_evictable=": "mem_cache/memory_pool.py",
    "mamba_protected=": "mem_cache/memory_pool.py",
    "pinned checkpoint": "mem_cache/mamba_pool_floor.py",
}

#: The barlink half greps these out of its own transport.
BARLINK_MARKERS = {
    "DeviceCollectiveAborted": (
        "distributed/device_communicators/barlink_device.py"
    ),
}

#: Asserts that killed the 2026-08-05 arms. #581 must keep them unreachable,
#: but the STRINGS must survive: the harness proves their absence, and you
#: cannot prove the absence of a string that no longer exists anywhere.
REMOVED_DEATH_MARKERS = {
    "Not enough space for mamba ping pong idx": "mem_cache/memory_pool.py",
    "Not enough space for mamba cache": "mem_cache/memory_pool.py",
}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / _HARNESS).is_file():
            return parent
    raise AssertionError(f"could not locate {_HARNESS}")


class TestHarnessMarkersExistInSources(unittest.TestCase):
    def test_every_pinned_marker_is_verbatim_in_its_source(self):
        root = _repo_root()
        for literal, rel in PINNED_MARKERS.items():
            with self.subTest(marker=literal):
                text = (root / _SRT / rel).read_text()
                self.assertIn(
                    literal,
                    text,
                    f"the soak harness greps {literal!r} but {rel} no longer "
                    f"contains it. Update BOTH the harness and this pin, or "
                    f"the soak will silently report the #581 acceptance as "
                    f"'not observed'.",
                )

    def test_barlink_marker_is_verbatim_in_its_source(self):
        root = _repo_root()
        for literal, rel in BARLINK_MARKERS.items():
            with self.subTest(marker=literal):
                self.assertIn(literal, (root / _SRT / rel).read_text())

    def test_death_assert_strings_still_exist_to_be_proven_absent(self):
        """The harness proves these do not appear in a run's log.

        If the strings were deleted outright the harness would keep passing
        for the wrong reason. They must remain in the source (unreachable is
        fine) so that their absence from a LOG is meaningful evidence.
        """
        root = _repo_root()
        for literal, rel in REMOVED_DEATH_MARKERS.items():
            with self.subTest(marker=literal):
                self.assertIn(
                    literal,
                    (root / _SRT / rel).read_text(),
                    f"{literal!r} vanished from {rel}. The soak's #581 "
                    f"acceptance greps for it; with the string gone the "
                    f"acceptance passes vacuously.",
                )


class TestHarnessGrepsOnlyPinnedMarkers(unittest.TestCase):
    """Every fixed-string grep in the harness must be a declared pin.

    This is the direction that actually rots: someone adds a `grep -cF` for a
    new marker, never registers it here, and it quietly stops matching.
    """

    def test_no_unregistered_fixed_string_greps(self):
        root = _repo_root()
        harness = (root / _HARNESS).read_text()
        found = set(re.findall(r'grep -c?F "([^"]+)"', harness))
        known = set(PINNED_MARKERS) | set(BARLINK_MARKERS) | set(
            REMOVED_DEATH_MARKERS
        )
        unregistered = {f for f in found if f not in known}
        self.assertFalse(
            unregistered,
            f"the harness greps these fixed strings without a pin here: "
            f"{sorted(unregistered)}. Add them to PINNED_MARKERS with the "
            f"source file that must contain them.",
        )


if __name__ == "__main__":
    unittest.main()
