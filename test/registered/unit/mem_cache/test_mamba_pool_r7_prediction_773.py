"""#773: the derived pool size, driven through the REAL sizing entry point.

WHY THIS FILE EXISTS. The sizing tests in
`model_executor/test_mamba_pool_from_floor_773.py` call
`_auto_mamba_demand_size(ratio)` with a ratio passed BY HAND. That is how the
ratio defect shipped: the hand-passed 2 proved a number the call site never
computes, while `handle_max_mamba_cache` multiplied by
`_calculate_mamba_ratio()` = 3 and would have derived 30 -- larger than the
hand-pinned 24 it was meant to replace. A test that picks its own inputs
cannot catch that. This one drives `handle_max_mamba_cache` itself, bound
from the real class, and reads the size the production path actually
installs.

IT ALSO PINS A COUPLING THAT IS EASY TO MISS. Dropping
`--max-mamba-cache-size` is only an improvement while the #755 reorder ARMS.
If it does not, the floor is 1+P+1+1 rather than 1+P+1, the capped ratio
follows it back up to 3, and the derived pool becomes 30 -- a REGRESSION of
225 MiB per rank against the pin it replaced. So the two changes are not
independent, and the boot that removes the pin must confirm `floor=16` in the
MAMBA-FLOOR line before trusting the pool it gets.

Numbers are the standing boot's binding rank (PP0, boot_798_0822_0646):
0.877 GiB of main mamba state over 24 slots = 37.42 MiB per request, with
12.153 GiB of post-weights budget left.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_mamba_checkpoint_interval import _FakeServerArgs, _make_mock_runner

register_cpu_ci(est_time=10)

GIB = 1 << 30
MIB = 1 << 20

#: PP0's per-request mamba state on the standing boot.
PER_REQ = int(0.877 * GIB / 24)
#: PP0's post-weights budget on the same boot ("rest=" in the KV budget line).
REST_GB = 12.153
#: What the boot pins today.
PINNED = 24


def _size(pin, reorder_on: bool) -> int:
    """Run the production sizing path and report the pool it installs."""
    prev = os.environ.get("SGLANG_MAMBA_SLOT_REORDER")
    os.environ["SGLANG_MAMBA_SLOT_REORDER"] = "1" if reorder_on else "0"
    try:
        sa = _FakeServerArgs(
            max_running_requests=8,
            speculative_num_draft_tokens=3,
            disable_radix_cache=False,
            dcp_size=1,
        )
        # --max-running-requests 8 is STATED on this boot, which is what lets
        # the demand path be reached at all (#773).
        sa.max_running_requests_user_set = True
        sa.enable_hierarchical_cache = True
        sa.hicache_write_policy = "write_through"
        sa.mamba_radix_cache_strategy = "no_buffer"
        sa.disable_overlap_schedule = True
        sa.max_mamba_cache_size = pin

        runner = _make_mock_runner(PER_REQ, sa, has_spec=True)
        runner.pp_size = 1  # PER_REQ is already this stage's local value
        runner.handle_max_mamba_cache(REST_GB)
        return sa.max_mamba_cache_size
    finally:
        if prev is None:
            os.environ.pop("SGLANG_MAMBA_SLOT_REORDER", None)
        else:
            os.environ["SGLANG_MAMBA_SLOT_REORDER"] = prev


class TestTheDerivedPoolThroughTheRealPath(CustomTestCase):
    def test_the_pin_is_still_obeyed_when_present(self):
        """An operator who names a size is not second-guessed."""
        self.assertEqual(_size(PINNED, reorder_on=True), PINNED)

    def test_dropping_the_pin_derives_twenty_and_costs_less(self):
        """THE PREDICTION the pin-removal boot is judged against."""
        size = _size(None, reorder_on=True)
        self.assertEqual(size, 20)
        freed_mib = (PINNED - size) * PER_REQ / MIB
        self.assertGreater(freed_mib, 140)
        self.assertLess(freed_mib, 160)

    def test_CAN_FAIL_without_the_reorder_dropping_the_pin_REGRESSES(self):
        """The coupling, stated as a test rather than left as a footnote.

        If the reorder does not arm, the floor rises to 1+P+1+1, the capped
        ratio follows, and the derived pool overshoots the pin it replaced.
        Removing the pin is therefore conditional on the reorder, and a boot
        must read `floor=16` off the MAMBA-FLOOR line before trusting the
        pool it is given.
        """
        size = _size(None, reorder_on=False)
        self.assertEqual(size, 30)
        self.assertGreater(
            size,
            PINNED,
            "unarmed, the derived pool is WORSE than the hand-pin -- which is "
            "why the two changes must not be judged independently",
        )

    def test_the_two_arms_differ_only_in_the_reorder(self):
        """Same inputs, same call, one env: the delta is attributable."""
        self.assertNotEqual(_size(None, reorder_on=True), _size(None, reorder_on=False))


if __name__ == "__main__":
    unittest.main()
