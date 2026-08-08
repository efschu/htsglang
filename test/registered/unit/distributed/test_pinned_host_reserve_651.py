"""#651: the pinned-host reserve must be settable per host.

The reserve exists because pinned pages cannot be swapped, so over-committing
them invokes the OOM killer rather than degrading. Its 10 GiB default was
sized for the rig, where 10 GiB is a small slice and there is no swap at all.

On the gfx1103 laptop that default is not merely conservative, it is wrong in
kind: total RAM is 29.5 GiB, the GGUF weights hold ~22.7 GiB of it through
GTT, and the machine does have swap for everything that is not pinned. A fixed
10 GiB reserve therefore refuses a 0.15 GB HiCache staging tier that the host
can trivially afford -- the refusal observed at boot as

    Pinned host RAM over-committed: 0.15 GB requested ... does not fit in
    1.69 GB available minus a 10.74 GB OS reserve = 0.00 GB usable.

These tests pin the override's behaviour, and in particular that a MALFORMED
setting can never turn the guard off -- the failure direction has to be the
conservative one, because the alternative failure is an OOM kill.
"""

import os
import unittest
from unittest import mock

from sglang.srt.mem_cache.pinned_host_budget import (
    PINNED_HOST_RESERVE_BYTES,
    PINNED_HOST_RESERVE_ENV,
    PinnedHostPost,
    joint_pinned_host_error,
    pinned_host_reserve_bytes,
)

_GIB = 1024**3
_MIB = 1024**2


def _env(value):
    """Environment with the reserve variable set to ``value`` (or removed)."""
    patch = {} if value is None else {PINNED_HOST_RESERVE_ENV: value}
    env = {k: v for k, v in os.environ.items() if k != PINNED_HOST_RESERVE_ENV}
    env.update(patch)
    return mock.patch.dict(os.environ, env, clear=True)


class TestPinnedHostReserveOverride(unittest.TestCase):
    def test_default_is_unchanged_when_unset(self):
        """An unset variable must leave the rig's behaviour exactly as it was."""
        with _env(None):
            self.assertEqual(pinned_host_reserve_bytes(), PINNED_HOST_RESERVE_BYTES)

    def test_empty_and_whitespace_are_treated_as_unset(self):
        for raw in ("", "   "):
            with self.subTest(raw=raw), _env(raw):
                self.assertEqual(pinned_host_reserve_bytes(), PINNED_HOST_RESERVE_BYTES)

    def test_override_is_read_in_mib(self):
        with _env("1024"):
            self.assertEqual(pinned_host_reserve_bytes(), 1024 * _MIB)

    def test_zero_is_honoured_as_an_explicit_opt_out(self):
        """Zero is a deliberate operator decision, not a malformed value."""
        with _env("0"):
            self.assertEqual(pinned_host_reserve_bytes(), 0)

    def test_malformed_value_falls_back_to_the_default(self):
        """A typo must not silently disable the guard."""
        for raw in ("abc", "1.5", "10GiB", "--"):
            with self.subTest(raw=raw), _env(raw):
                self.assertEqual(pinned_host_reserve_bytes(), PINNED_HOST_RESERVE_BYTES)

    def test_negative_value_falls_back_to_the_default(self):
        """A negative reserve would ADD headroom that does not exist."""
        with _env("-4096"):
            self.assertEqual(pinned_host_reserve_bytes(), PINNED_HOST_RESERVE_BYTES)

    def test_resolved_per_call_not_at_import(self):
        """The launcher may set the variable after this module is imported."""
        with _env("2048"):
            first = pinned_host_reserve_bytes()
        with _env("4096"):
            second = pinned_host_reserve_bytes()
        self.assertEqual(first, 2048 * _MIB)
        self.assertEqual(second, 4096 * _MIB)


class TestJointErrorUsesTheOverride(unittest.TestCase):
    """The laptop case end to end: same numbers, opposite verdict."""

    #: The boot that produced the refusal quoted in the module docstring.
    LAPTOP_POSTS = [
        PinnedHostPost(
            name="MHATokenToKVPoolHost",
            flag="--hicache-size / --hicache-ratio",
            nbytes=150_000_000,
        )
    ]
    LAPTOP_TOTAL = 31_664_000_000
    LAPTOP_AVAILABLE = 1_690_000_000

    def test_default_reserve_refuses_the_laptop_configuration(self):
        with _env(None):
            err = joint_pinned_host_error(
                self.LAPTOP_POSTS, self.LAPTOP_TOTAL, self.LAPTOP_AVAILABLE
            )
        self.assertIsNotNone(err)
        self.assertIn("OS reserve", err)
        # The refusal must still name the flag an operator would lower.
        self.assertIn("--hicache-size", err)

    def test_lowered_reserve_admits_the_laptop_configuration(self):
        with _env("1024"):
            err = joint_pinned_host_error(
                self.LAPTOP_POSTS, self.LAPTOP_TOTAL, self.LAPTOP_AVAILABLE
            )
        self.assertIsNone(err)

    def test_override_cannot_defeat_the_total_ram_ceiling(self):
        """Exceeding TOTAL is impossible on any machine state, reserve or not.

        This is the bound that keeps the knob honest: it can trade away the
        safety margin, but it cannot conjure RAM that does not exist.
        """
        posts = [
            PinnedHostPost(
                name="MHATokenToKVPoolHost",
                flag="--hicache-size",
                nbytes=64 * _GIB,
            )
        ]
        with _env("0"):
            err = joint_pinned_host_error(posts, 32 * _GIB, 30 * _GIB)
        self.assertIsNotNone(err)
        self.assertIn("TOTAL host RAM", err)

    def test_explicit_reserve_argument_still_wins_over_the_environment(self):
        """Callers that pass a reserve (the existing tests do) are unaffected."""
        with _env("0"):
            err = joint_pinned_host_error(
                self.LAPTOP_POSTS,
                self.LAPTOP_TOTAL,
                self.LAPTOP_AVAILABLE,
                10 * _GIB,
            )
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
