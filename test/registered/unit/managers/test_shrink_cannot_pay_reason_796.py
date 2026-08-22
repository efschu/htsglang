# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#796: the "cannot pay" refusal must report MEASURED state, not two guesses.

WHY THIS IS NOT COSMETIC. Three times on 2026-08-22 a diagnostic rather than a
mechanism set this chain's direction, and this line is the clearest instance
left. It names two causes -- "the arena has no commit chunk, or its handles are
retained (SGLANG_FLIP_SEAM_RETAIN_HANDLES)" -- separates them in neither code
nor fact, and on boot_798_0822_0810.log is wrong about both:

  * A CHUNKLESS ARENA CANNOT REACH THIS LINE. Registration refuses outright
    when the pool does not report ``supports_backing_spans``
    (kv_backing_relief.py, the "NO COMMIT CHUNK" gate), so every rung that
    exists has a chunk. Offering that as a candidate sends the reader to a
    branch the constructor already excluded.
  * RETENTION IS MEASURED, NOT GUESSED. ``arena_census()`` already reports
    ``retained`` bytes per device, allocation-free and read-only. Naming an
    env var the reader then has to go and check is strictly worse than
    printing the number.

AND THE THIRD STATE, the one the message does not have a branch for at all and
the one that actually fired: ``runtime_set_backing_rows`` returns BYTES
RELEASED TO THE DRIVER (memory_pool.py:4009-4012). On that boot all 24
refusals reported ``claimed=0``. Pool and driver AGREED -- there was no
divergence between "reported MiB" and "the driver's free column" to explain,
because nothing was ever reported. The message's whole story ("unmapping
without releasing yields address space rather than memory") presupposes an
unmap that did not happen. Meanwhile 15 shrinks on the SAME boot released
256/512 MiB, so neither standing cause can be the explanation for either set.

Hermetic: no CUDA, no scheduler, no distributed. The rung carries only the
fields ``_shrink_to`` reads (the #717 fixture idiom).
"""

from __future__ import annotations

import logging
import unittest
from unittest import mock

from sglang.srt.managers import kv_backing_relief as kbr

ROW_BYTES = 32_768
BUFFERS = 28
CHUNK_BYTES = 256 * 1024 * 1024

#: One commit chunk in EVERY buffer, in rows: 256 MiB x 28 / 32 KiB = 229376.
#: Worth stating as a number because it is startlingly large -- on this
#: geometry an ask has to exceed 224 Ki ROWS before it can release a single
#: byte, which is more rows than several plausible pools hold in total.
GRANULARITY_ROWS = CHUNK_BYTES * BUFFERS // ROW_BYTES

CURRENT = 1_000_000
#: Deep enough that ``asked`` clears ``_min_release_rows`` comfortably, so a
#: zero release is never explainable as "the ask was below granularity".
TARGET = 700_000


class _Cap:
    def __init__(self):
        self.engaged = None
        self.released = 0

    def engage(self, target):
        self.engaged = target

    def release(self):
        self.released += 1


def _relief(*, claimed_bytes: int, chunk_bytes: int = CHUNK_BYTES):
    """A rung carrying only what ``_shrink_to`` reads."""
    r = kbr.KvBackingRelief.__new__(kbr.KvBackingRelief)

    pool = type("P", (), {})()
    pool.page_size = 1
    pool.backing_commit_chunk_bytes = chunk_bytes
    pool.runtime_set_backing_rows = lambda rows: claimed_bytes
    r._pool = pool

    r._bytes_per_row = ROW_BYTES
    r._buffers = BUFFERS
    r._device = 0
    r._device_index = 0
    # The driver's free column never moves in any of these cases: that is the
    # condition under test, not an incidental.
    r._probe = lambda: 4_000 * 1024 * 1024
    r._rows_at_boot = CURRENT
    r._cap = _Cap()
    r.shrink_count = 0
    r.released_total = 0
    r._exhausted_at_rows = None
    r._exhausted_target_rows = None

    # The eviction leg is not what this test is about: let it succeed and put
    # the live set far below the target so the #717 floor never raises the cap.
    r._lower_watermark_to = lambda target: 0
    r._max_live_row = lambda: 1_000
    r._floor_rows = lambda live: live + 512
    r._current_rows = lambda: CURRENT
    return r


def _warn(r) -> str:
    logger = logging.getLogger(kbr.__name__)
    with mock.patch.object(logger, "warning") as w:
        got = r._shrink_to(TARGET, CURRENT)
    assert got == 0, "a pool that released nothing must report zero"
    calls = [c for c in w.call_args_list if "cannot pay" in str(c.args[0]).lower()
             or "did not move" in str(c.args[0]).lower()
             or "released nothing" in str(c.args[0]).lower()]
    assert calls, f"the refusal must warn; got {w.call_args_list!r}"
    fmt, *rest = calls[-1].args
    return fmt % tuple(rest)


class TestTheRefusalReportsMeasuredState(unittest.TestCase):
    def test_a_pool_that_claimed_nothing_is_not_reported_as_a_divergence(self):
        """claimed == 0: pool and driver AGREE. There is nothing to reconcile.

        This is the branch that fired 24 times on metal. Saying the shrink
        "reported N MiB but the driver's free column did not move" when N is
        zero describes a disagreement that does not exist, and points the
        reader at unmap semantics instead of at the pool's own refusal.
        """
        msg = _warn(_relief(claimed_bytes=0)).lower()
        self.assertNotIn(
            "but the driver's free column did not move",
            msg,
            "with claimed=0 there is no pool-vs-driver divergence to report; "
            f"got {msg!r}",
        )
        self.assertIn(
            "claimed=0",
            msg.replace(" ", ""),
            "the deciding number must be in the line, not inferable from it",
        )

    def test_the_chunkless_candidate_is_not_offered_when_the_chunk_is_known(self):
        """Registration already excluded it, and the value is right there."""
        msg = _warn(_relief(claimed_bytes=0)).lower()
        self.assertNotIn(
            "no commit chunk",
            msg,
            "the constructor's supports_backing_spans gate refuses to register "
            "a chunkless arena, so this candidate is dead on arrival here",
        )
        self.assertIn(
            "256",
            msg,
            "print the measured commit chunk in MiB instead of hypothesising "
            f"about its absence; got {msg!r}",
        )

    def test_retention_is_reported_as_a_number_not_as_an_env_var(self):
        with mock.patch(
            "sglang.srt.mem_cache.kv_vmm_backing.arena_census",
            return_value={0: {"reserved": 8 << 30, "backed": 6 << 30,
                              "retained": 1536 * 1024 * 1024, "arenas": 2}},
        ):
            msg = _warn(_relief(claimed_bytes=0))
        self.assertIn(
            "1536",
            msg,
            f"arena_census() measures retention; print it. Got {msg!r}",
        )
        self.assertNotIn(
            "SGLANG_FLIP_SEAM_RETAIN_HANDLES",
            msg,
            "naming an env var the reader must go and check is strictly worse "
            "than printing the number it controls",
        )

    def test_an_unreadable_census_says_unknown_rather_than_inventing_one(self):
        """A census that cannot be read must not become a confident zero."""
        with mock.patch(
            "sglang.srt.mem_cache.kv_vmm_backing.arena_census",
            side_effect=RuntimeError("registry gone"),
        ):
            msg = _warn(_relief(claimed_bytes=0)).lower()
        self.assertIn("unknown", msg, f"got {msg!r}")

    def test_a_real_divergence_is_still_reported_as_one(self):
        """claimed > 0 with a flat free column IS the address-space story.

        The branch must survive: it is a real failure mode, it just was not
        this boot's.
        """
        msg = _warn(_relief(claimed_bytes=512 * 1024 * 1024)).lower()
        self.assertIn("512", msg)
        self.assertIn(
            "address space",
            msg,
            "when the pool DID report bytes and the driver did not see them, "
            f"the unmap-without-release reading is the right one; got {msg!r}",
        )


class TestTheLadderDoesNotMove(unittest.TestCase):
    """This change is instrumentation. It must not alter a single decision."""

    def test_the_cap_stays_on_and_nothing_is_recovered(self):
        r = _relief(claimed_bytes=0)
        _warn(r)
        self.assertEqual(r._cap.engaged, TARGET, "the cap must stay engaged")
        self.assertEqual(r._cap.released, 0, "undoing the cap re-commits pages")
        self.assertEqual(r.shrink_count, 0)
        self.assertEqual(r.released_total, 0)

    def test_exhaustion_is_still_marked_when_the_ask_cleared_granularity(self):
        self.assertGreater(
            CURRENT - TARGET, GRANULARITY_ROWS, "fixture must clear granularity"
        )
        r = _relief(claimed_bytes=0)
        _warn(r)
        self.assertEqual(
            r._exhausted_at_rows,
            CURRENT,
            "an ask above one commit chunk per buffer that returned nothing is "
            "still evidence of exhaustion; only the WORDING changes here",
        )

    def test_an_ask_below_granularity_is_not_counted_as_exhaustion(self):
        """The rule the refusal line now PRINTS, asserted rather than assumed.

        Below one commit chunk per buffer a zero release is arithmetic, not
        evidence about the arena. Marking it exhausted would silence a rank
        that has real bytes to offer at a deeper target. This test exists
        because writing the granularity into the message is only useful if the
        message's own claim is true.
        """
        shallow = CURRENT - (GRANULARITY_ROWS // 2)
        r = _relief(claimed_bytes=0)
        logger = logging.getLogger(kbr.__name__)
        with mock.patch.object(logger, "warning"):
            r._shrink_to(shallow, CURRENT)
        self.assertIsNone(
            r._exhausted_at_rows,
            "an ask smaller than this rank's own release granularity says "
            "nothing about the arena; it says the group agreed on a number "
            "this rank cannot act on",
        )


if __name__ == "__main__":
    unittest.main()
