"""#814: the KV cap must not outlive the flip it helped refuse.

`recover_kv_backing` is the ONLY thing that lifts `KvRowCap`, and until this
change its only call sites were the two post-cutover hooks
(phase_flip_runtime.py:1818 and :1835). That is a trap the moment a flip stops
committing, and the trap closes on itself:

    corridor-bounded recovery leaves the ranks unequal (measured: 210944 /
    124928 / 131072 backed rows) -> the cap agreement levels the group to the
    poorest, 124928 of 465190 -> the capped pool fills, so the live floor is
    100% of what is left -> the next flip is DECLINED for want of fit -> no
    cutover -> no recovery -> the cap is never lifted -> repeat.

Measured shape of that trap in one boot: 27 pool censuses, three ranks, nine
events, and NOT ONE `post-cutover tp_to_pp`. All six return attempts stopped at
`at-arm`; `recovered to N of M rows` appears zero times; a user got an
overloaded_error against a pool that had been sized 3.7x larger.

`recover_kv_backing_on_abandon` adds the missing leg at the abandon exit, where
nothing moved and no seam is owed the memory. These tests drive that helper
directly -- no GPU, no flip, no scheduler.

WHAT IS NOT TESTED HERE, stated rather than implied: that the call site is
reached by every rank. That is a property of `_execute`'s control flow (the
exit is downstream of a bit-identical `_collective_min` with no return and no
raise in between), checked by reading it, and asserted at boot by the tree's
file:line presence gate -- not by this file.
"""

import logging
import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.managers import phase_flip_spill as pfs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)

LOGGER = "sglang.srt.managers.phase_flip_spill"


def _reduce(payload):
    """A stand-in MIN channel. Its identity is what the helper must pass on."""
    return list(payload)


def _call(reduce_fn=_reduce, delegate=None, direction="tp_to_pp"):
    """Run the helper with `recover_kv_backing` replaced, capturing the logs."""
    scheduler = SimpleNamespace()
    with unittest.mock.patch.object(
        pfs, "recover_kv_backing", delegate or unittest.mock.Mock(return_value=0)
    ) as inner:
        with unittest.mock.patch.object(
            logging.getLogger(LOGGER), "warning"
        ) as warn, unittest.mock.patch.object(
            logging.getLogger(LOGGER), "error"
        ) as err:
            got = pfs.recover_kv_backing_on_abandon(
                scheduler, reduce_fn, direction=direction, why="seam fit refused"
            )

    def _text(mock):
        if not mock.called:
            return ""
        a = mock.call_args[0]
        return a[0] % tuple(a[1:])

    return got, inner, _text(warn), _text(err)


class TestItLiftsTheCapAtAnAbandon(CustomTestCase):
    def test_it_delegates_through_the_group_channel(self):
        """The channel must be PASSED ON, not swallowed.

        The levelling half of the recovery is collective; a delegate called
        without `reduce_fn` would grow rank-locally and leave the ranks on
        different id spaces, which is the one state the seam cannot repair.
        """
        got, inner, _, _ = _call(delegate=unittest.mock.Mock(return_value=4096))
        self.assertEqual(got, 4096)
        self.assertTrue(inner.called)
        self.assertIs(inner.call_args.kwargs["reduce_fn"], _reduce)

    def test_a_real_recovery_is_announced_with_its_row_count(self):
        _, _, warn, _ = _call(delegate=unittest.mock.Mock(return_value=4096))
        self.assertIn("4096", warn)
        self.assertIn("ABANDONED", warn)


class TestItCannotTurnALostFlipIntoADeadRank(CustomTestCase):
    """An abandon is already the unhappy path. It must stay survivable."""

    def test_a_raising_recovery_is_contained(self):
        def boom(*a, **k):
            raise RuntimeError("cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY")

        got, _, _, err = _call(delegate=boom)
        self.assertEqual(got, 0)
        self.assertIn("cuMemCreate", err)
        self.assertIn("capacity loss, never a fault", err)


class TestItIsSilentWhenThereIsNothingToSay(CustomTestCase):
    """Abandons come in floods -- 47 in a row were measured on this rig."""

    def test_a_zero_row_recovery_logs_nothing(self):
        _, _, warn, err = _call(delegate=unittest.mock.Mock(return_value=0))
        self.assertEqual(warn, "")
        self.assertEqual(err, "")


class TestSingleRankShapesAreLeftAlone(CustomTestCase):
    def test_without_a_channel_it_is_a_no_op(self):
        """AND IT MUST NOT REACH THE POOL AT ALL.

        No channel means no group agreement is possible, so a grow here could
        only ever be rank-local -- exactly the divergence the levelling exists
        to prevent. Returning early is not an optimisation; it is the contract
        `recover_kv_backing` already keeps for the same case.
        """
        got, inner, warn, err = _call(reduce_fn=None)
        self.assertEqual(got, 0)
        self.assertFalse(inner.called)
        self.assertEqual(warn, "")
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
