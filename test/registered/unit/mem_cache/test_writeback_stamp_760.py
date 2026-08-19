# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#760: a host write-back must be refused if the binding moved under it.

WHY A SECOND CHECK IS NEEDED. The write path already refuses at ENQUEUE time
(`device_tier_disarmed("write")`). It does not re-check at CONSUME time, and
that is the window both crash specimens died in: each took a SIGSEGV exactly
three seconds after a `pp_to_tp` cutover completed (14:08:14 -> 14:08:17,
epoch 27; 07:12:09 -> 07:12:12, epoch 3, seven hours apart), inside
`MHATokenToKVPoolHost.backup_from_device_all_layer`.

WHY THE SHAPE GUARD CANNOT COVER IT. Under `layer_first` the host layout
equals the device layout, so a stale binding is shape-IDENTICAL to the live
one. `check_shapes` passes by construction -- which is what the #760 record
shows: the guard armed on all three ranks, refused zero transfers, and the
process segfaulted anyway. Matching shapes plus a crash put the fault below
the Python seam; a generation stamp is what distinguishes "same shape" from
"same pool".
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache import hicache_phase_binding as pb


class TestAStampSurvivesOnlyItsOwnGeneration(unittest.TestCase):
    def test_a_fresh_stamp_is_current(self):
        g = pb.current_generation()
        self.assertTrue(pb.write_back_stamp_is_current(g))

    def test_a_stamp_from_before_a_rebind_is_refused(self):
        stamped = pb.current_generation()
        pb.binding_state().advance("tp")  # the cutover happens here
        self.assertFalse(
            pb.write_back_stamp_is_current(stamped),
            "a write-back queued before the rebind must not be consumed after it",
        )

    def test_an_unstamped_write_back_is_refused(self):
        # No stamp means no evidence it belongs to this binding.
        self.assertFalse(pb.write_back_stamp_is_current(None))

    def test_the_new_generation_is_accepted_again(self):
        pb.binding_state().advance("pp")
        self.assertTrue(pb.write_back_stamp_is_current(pb.current_generation()))


if __name__ == "__main__":
    unittest.main()
