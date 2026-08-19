# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#767: the #755 early release may not fire against an IN-FLIGHT host copy.

WHAT WENT WRONG. ``mamba_backuped`` reads ``mamba_host_value is not None``, and
the write-through path publishes that value in the same block that hands the
transfer to the cache controller and records the node in
``ongoing_write_through``. So the anchor read as "host-backed" while its bytes
were still in flight. #755 released the pin there, the node became evictable
before its copy existed, and a resume off that dead anchor produced degenerate
text -- 9 of 10 salted greedy "capital of France" probes on the full boot, and
the same prompt answered correctly on the 3rd try, which is what an
intermittently missing anchor looks like from outside.

The predicate's own docstring already promised "host-backed RIGHT NOW"; these
tests make that literal.
"""

from __future__ import annotations

import unittest


class _Node:
    def __init__(self, node_id=1, host_value=object()):
        self.id = node_id
        self.mamba_host_value = host_value

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None


class _Base:
    """The device-only pool: no async write-through."""

    mamba_slot_reorder = True
    root_node = None

    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

    _mamba_early_release_admissible = MambaRadixCache._mamba_early_release_admissible
    _mamba_host_copy_complete = MambaRadixCache._mamba_host_copy_complete


class _Hier(_Base):
    """The hierarchical pool: publishes the value when the copy is QUEUED."""

    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

    _mamba_host_copy_complete = HiMambaRadixCache._mamba_host_copy_complete

    def __init__(self):
        self.ongoing_write_through = {}
        self._write_through_inflight = {}


class TestTheDeviceOnlyPoolIsUnchanged(unittest.TestCase):
    def test_a_backed_node_is_admissible(self):
        self.assertTrue(_Base()._mamba_early_release_admissible(_Node()))

    def test_an_unbacked_node_is_not(self):
        self.assertFalse(
            _Base()._mamba_early_release_admissible(_Node(host_value=None))
        )

    def test_the_reorder_switch_still_gates_everything(self):
        base = _Base()
        base.mamba_slot_reorder = False
        self.assertFalse(base._mamba_early_release_admissible(_Node()))


class TestAnInFlightCopyIsNotAnAnchor(unittest.TestCase):
    """The #767 defect, pinned. Each of these was True before the fix."""

    def test_a_node_with_an_inflight_write_is_refused(self):
        h = _Hier()
        node = _Node(node_id=7)
        h._write_through_inflight[7] = 1
        self.assertTrue(node.mamba_backuped, "precondition: it LOOKS backed")
        self.assertFalse(h._mamba_early_release_admissible(node))

    def test_a_node_still_in_ongoing_write_through_is_refused(self):
        h = _Hier()
        node = _Node(node_id=9)
        h.ongoing_write_through[9] = node
        self.assertFalse(h._mamba_early_release_admissible(node))

    def test_once_the_write_is_acked_the_release_is_admissible(self):
        h = _Hier()
        node = _Node(node_id=11)
        h._write_through_inflight[11] = 1
        h.ongoing_write_through[11] = node
        self.assertFalse(h._mamba_early_release_admissible(node))
        # the ack path clears both
        del h._write_through_inflight[11]
        del h.ongoing_write_through[11]
        self.assertTrue(h._mamba_early_release_admissible(node))

    def test_an_unbacked_node_is_refused_even_with_no_inflight_write(self):
        h = _Hier()
        self.assertFalse(
            h._mamba_early_release_admissible(_Node(node_id=13, host_value=None))
        )


if __name__ == "__main__":
    unittest.main()
