# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#624: the stub-drift audit. See barlink_stub_support for the mechanism.

RED-FIRST RECORD. The first run of ``test_stub_covers_or_excludes_every_
init_attr`` was executed with an EMPTY-but-for-examples exclusion table and
went red naming every uncovered attribute (~70 names) — that red run is the
attributable demonstration of the drift #624 describes. The exclusion table
was then populated from that output, each entry reviewed against the abort
surface (check_aborted / _read_status_for_check / poll_status_word / the
Bar1CollectiveAborted raise path).
"""

import unittest

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
)

from barlink_stub_support import ABORT_STUB_EXCLUSIONS, init_assigned_attrs
from test_barlink_bar1_abort_deferred_517 import _transport


class TestBarlinkStubDrift624(unittest.TestCase):
    def test_stub_covers_or_excludes_every_init_attr(self):
        """Every __init__ attribute is stub-set or consciously excluded."""
        real = init_assigned_attrs(BarlinkBar1Transport)
        stub = set(_transport().__dict__)
        excluded = set(ABORT_STUB_EXCLUSIONS)
        missing = sorted(real - stub - excluded)
        self.assertEqual(
            missing,
            [],
            "NEW transport attribute(s) not covered by the abort stub and "
            "not consciously excluded — decide per name (add to _transport "
            f"or to ABORT_STUB_EXCLUSIONS with a reason): {missing}",
        )

    def test_no_stale_exclusions(self):
        """An exclusion for an attribute __init__ no longer assigns is dead
        weight that would mask a future rename — the comparison is exact in
        both directions."""
        real = init_assigned_attrs(BarlinkBar1Transport)
        stale = sorted(set(ABORT_STUB_EXCLUSIONS) - real)
        self.assertEqual(stale, [], f"stale exclusions: {stale}")

    def test_no_double_bookkeeping(self):
        """An attribute both stub-set and excluded means the table lies."""
        stub = set(_transport().__dict__)
        both = sorted(stub & set(ABORT_STUB_EXCLUSIONS))
        self.assertEqual(both, [], f"attrs both set and excluded: {both}")


if __name__ == "__main__":
    unittest.main()
