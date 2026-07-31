"""all_reduce and all_to_all in rounds -- and the loud fallback notice.

Context, from the #293 wait analysis of the s12 logs. The standard run's
prefill all_reduce sits at a constant 20.0 MiB: 2048 tokens x 5120 x 2 byte,
shard 6.67 MiB per rank against a slot of 7.996 MiB. One round, 20 % headroom.
The tipping point is a payload above ``3 x 8188 KiB``, i.e. **above 2456 tokens
per batch** -- and above it ``handles()`` said False and the payload fell back
to the base transport without a single line saying so. ``chunked_prefill_size``
4096 or 8192, both usual in sglang, would have switched the direct path off in
prefill silently.

all_gather and broadcast have had ceil rounds for a while (``ag_plan`` /
``bc_plan``, group-uniform, ragged tail). The all_reduce and all_to_all paths
did not. They do now, on the same pattern and with no new kernel: a round is a
complete collective over a slice, the round count falls out of size and slot
alone, and a slice that ends early rides the remaining rounds with length 0.

CPU-only. The decomposition is a pure function, the gates are checked against a
stub, and the byte movement is replayed in Python against a reference.
"""

import logging
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
    a2a_rounds,
    ar_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The geometry of the run, from the setup line: region 96.0 MiB per rank,
#: 12 slots, slot 8188 KiB, largest payload 24564 KiB.
CHUNK_MAX = 8188 * 1024
SLOT = CHUNK_MAX
WORLD = 3
#: Hidden width and element size of the model -- so the ladder can be read
#: in TOKENS, the way the analysis writes it.
HIDDEN, ELEM = 5120, 2
#: The tipping point, computed: payload > 3 x 8188 KiB.
TIPPING_TOKENS = 2456


def _bytes(token: int) -> int:
    return token * HIDDEN * ELEM


def _stub(rank=0, world=WORLD, **kw):
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.world = world
    t.rank = rank
    t._up = True
    t._ext = object()
    t._proofs_hold = True
    t._a2a_proof = True
    t._bc_proof = True
    t.a2a_on = True
    t.ag_on = True
    t.bc_on = True
    t.min_bytes = 4096
    t.max_bytes = 24564 * 1024
    t.a2a_min_bytes = 16
    t.ag_min_bytes = 1
    t.bc_min_bytes = 1
    t.ag_max_rounds = 16
    t.bc_max_rounds = 16
    t.ar_max_rounds = 16
    t.a2a_max_rounds = 16
    t.ring_from = 1 << 20
    t.pipe_on = False
    t.pipe_from = 256 << 10
    t._plan = None
    t._window_minimum = 96 << 20
    t._geo = {
        "off_a2a": 4096,
        "a2a_slot": SLOT,
        "chunk_max": CHUNK_MAX,
        "region_bytes": 90 << 20,
    }
    for k, v in kw.items():
        if k in ("off_a2a", "a2a_slot", "chunk_max", "region_bytes"):
            t._geo[k] = v
        else:
            setattr(t, k, v)
    return t


class TestArPlanArithmetic(CustomTestCase):
    """The decomposition. Every byte once, every round inside the slot."""

    def _check(self, nbytes, chunk_max=CHUNK_MAX, world=WORLD):
        plan = ar_plan(nbytes, chunk_max, world)
        seen = []
        for offset, length in plan:
            self.assertGreater(length, 0)
            self.assertEqual(length % 16, 0, msg="round is not a multiple of 16")
            self.assertEqual(offset % 16, 0, msg="offset is not aligned")
            # The host insists on a 128-bit packet PER RANK (TORCH_CHECK
            # n4 >= R). A remainder round below that would not be a slow
            # case, it would be an abort.
            self.assertGreaterEqual(
                length // 16, world, msg=f"round {length} B under one packet per rank"
            )
            shard = -(-(length // 16) // world) * 16
            self.assertLessEqual(shard, chunk_max, msg="shard does not fit")
            seen.extend(range(offset, offset + length))
        self.assertEqual(seen, list(range(nbytes)))
        return plan

    def test_the_working_point_stays_one_round(self):
        """2048 tokens, 20.0 MiB -- the measured working point.

        This must not change: the same single round as before, otherwise
        all the run's numbers would suddenly no longer be comparable.
        """
        self.assertEqual(len(self._check(_bytes(2048))), 1)

    def test_the_documented_tipping_point(self):
        """2456 tokens still fit, 2457 no longer do -- and now they run."""
        self.assertEqual(len(self._check(_bytes(TIPPING_TOKENS))), 1)
        self.assertEqual(len(self._check(_bytes(TIPPING_TOKENS + 1))), 2)

    def test_the_usual_chunked_prefill_sizes(self):
        """The two sizes at which bar1 would have silently sat out
        prefill."""
        self.assertEqual(len(self._check(_bytes(4096))), 2)
        self.assertEqual(len(self._check(_bytes(8192))), 4)

    def test_the_ladder_is_gapless(self):
        for token in (2048, 2456, 2457, 4096, 8192):
            self._check(_bytes(token))

    def test_rounds_are_evenly_distributed(self):
        """Don't fill to the brim: the tail would otherwise be arbitrarily
        small, and the host rejects a round under one packet per rank."""
        for nbytes in (_bytes(2457), _bytes(4096) + 16, 25153536 + 16):
            plan = self._check(nbytes)
            lengths = [length for _, length in plan]
            # At most one packet of difference between the largest and
            # smallest round.
            self.assertLessEqual(max(lengths) - min(lengths), 16, msg=str(nbytes))

    def test_a_tail_of_one_packet_cannot_happen(self):
        """The case the even distribution rules out.

        Filled to the brim, this payload would have produced a remainder
        round of 16 bytes -- one packet, which the host aborts on with
        three ranks.
        """
        plan = self._check(25153536 + 16)
        self.assertEqual(len(plan), 2)
        self.assertTrue(all(length // 16 >= WORLD for _, length in plan))

    def test_the_round_count_is_group_uniform(self):
        """The graph condition. It depends on nbytes, chunk_max and world --
        on nothing that differs per rank."""
        nbytes = _bytes(8192)
        counts = {len(ar_plan(nbytes, CHUNK_MAX, WORLD)) for _ in range(8)}
        self.assertEqual(counts, {4})

    def test_smallest_and_largest_slot(self):
        self.assertEqual(len(ar_plan(48, 16, 3)), 1)
        self.assertEqual(len(ar_plan(1 << 20, 1 << 30, 2)), 1)

    def test_rejects_nonsense(self):
        for bad in ((16, 0, 3), (16, CHUNK_MAX, 1), (-16, CHUNK_MAX, 3),
                         (17, CHUNK_MAX, 3)):
            with self.assertRaises(ValueError, msg=str(bad)):
                ar_plan(*bad)
        self.assertEqual(ar_plan(0, CHUNK_MAX, 3), [])


class TestArPlanAgainstAReference(CustomTestCase):
    """Replay the round loop and compare byte for byte.

    Models what ``barlink_all_reduce`` does: every round reduces its slice, the
    slices are written back into the result at their own offset. The
    reference is the element-wise sum over all ranks of the whole buffer.
    """

    def _simulate(self, nbytes, world, chunk_max):
        data = {
            r: [((r * 31 + i * 7) % 251) for i in range(nbytes // 4)]
            for r in range(world)
        }
        # 0xEE analogue: whatever no round touches stays recognizably wrong.
        result = [-1] * (nbytes // 4)
        for offset, length in ar_plan(nbytes, chunk_max, world):
            a, b = offset // 4, (offset + length) // 4
            for i in range(a, b):
                result[i] = sum(data[r][i] for r in range(world))
        expected = [sum(data[r][i] for r in range(world)) for i in range(nbytes // 4)]
        return result, expected

    def test_one_round(self):
        actual, expected = self._simulate(_bytes(2048), WORLD, CHUNK_MAX)
        self.assertEqual(actual, expected)

    def test_many_rounds(self):
        for token in (2457, 4096, 8192):
            actual, expected = self._simulate(_bytes(token), WORLD, CHUNK_MAX)
            self.assertEqual(actual, expected, msg=f"{token} tokens")

    def test_small_geometry_many_rounds(self):
        """Small numbers, so the round logic itself gets checked, not just
        the arithmetic of large multiples."""
        actual, expected = self._simulate(4096, 3, 64)
        self.assertEqual(actual, expected)

    def test_no_element_is_written_twice_or_skipped(self):
        for nbytes in (4096, 4096 + 16, 65536):
            plan = ar_plan(nbytes, 64, 3)
            touched = []
            for offset, length in plan:
                touched.extend(range(offset, offset + length))
            self.assertEqual(touched, list(range(nbytes)), msg=str(nbytes))


class TestA2aRounds(CustomTestCase):
    """The same answer for all_to_all, from the group-wide largest block."""

    def test_a_block_inside_the_slot_is_one_round(self):
        self.assertEqual(a2a_rounds(SLOT, SLOT), 1)
        self.assertEqual(a2a_rounds(1, SLOT), 1)

    def test_a_block_over_the_slot_is_split(self):
        self.assertEqual(a2a_rounds(SLOT + 1, SLOT), 2)
        self.assertEqual(a2a_rounds(SLOT * 4, SLOT), 4)

    def test_zero_is_still_one_round(self):
        """All blocks empty still means: the barrier still runs."""
        self.assertEqual(a2a_rounds(0, SLOT), 1)

    def test_the_count_comes_from_the_group_wide_maximum(self):
        """Computed from a rank's own row it would be rank-dependent -- and
        a rank with one fewer round is a hang, not a bug."""
        own_rows = [SLOT // 2, SLOT * 3, SLOT]
        group_wide = max(own_rows)
        counts = {a2a_rounds(group_wide, SLOT) for _ in own_rows}
        self.assertEqual(counts, {3})

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            a2a_rounds(16, 0)
        with self.assertRaises(ValueError):
            a2a_rounds(-1, 16)

    def test_the_gate_follows_the_round_cap(self):
        t = _stub(a2a_max_rounds=4, a2a_slot=1024)
        self.assertTrue(t.supports_a2a(1024 * 4))
        self.assertFalse(t.supports_a2a(1024 * 4 + 1))

    def test_the_transport_reports_the_count_for_the_seam(self):
        t = _stub(a2a_slot=1024)
        self.assertEqual(t.a2a_rounds_for(1024 * 3), 3)


class TestHandlesGate(CustomTestCase):
    """Coverage before and after, at the sizes the analysis names."""

    def test_the_working_point_is_covered_as_before(self):
        t = _stub()
        self.assertTrue(t.handles("all_reduce", _bytes(2048)))
        self.assertEqual(t.ar_rounds(_bytes(2048)), 1)

    def test_over_the_tipping_point_is_now_covered(self):
        """BEFORE: handles() -> False and a silent fallback."""
        t = _stub()
        for token in (TIPPING_TOKENS + 1, 4096, 8192):
            self.assertTrue(
                t.handles("all_reduce", _bytes(token)), msg=f"{token} Token"
            )

    def test_the_coverage_is_gapless_up_to_the_round_cap(self):
        t = _stub()
        per_round = (CHUNK_MAX // 16) * WORLD * 16
        self.assertTrue(t.handles("all_reduce", per_round * 16))
        self.assertFalse(t.handles("all_reduce", per_round * 16 + 16))

    def test_the_hard_host_conditions_still_reject(self):
        """What the host does not run, the seam does not run either."""
        t = _stub()
        self.assertFalse(t.handles("all_reduce", 4096 + 1))    # not a multiple of 16
        self.assertFalse(t.handles("all_reduce", 32))          # below min_bytes
        tiny = _stub(min_bytes=16)
        self.assertFalse(tiny.handles("all_reduce", 32))     # < one packet per rank

    def test_a_round_that_would_not_fit_the_window_is_refused(self):
        t = _stub(_window_minimum=1 << 20)
        self.assertFalse(t.handles("all_reduce", _bytes(2048)))

    def test_the_other_ops_are_unaffected(self):
        t = _stub()
        self.assertFalse(t.handles("reduce_scatter", 65536))
        self.assertTrue(t.handles("all_gather", 65536))
        self.assertTrue(t.handles("broadcast", 128))


class TestLoudFallbackNotice(CustomTestCase):
    """Falling back is allowed, doing it silently is not.

    Outside a capture the gloo plane is a slower but usable path, so nothing
    aborts. But a transport that quietly steps aside for one size while the
    log says "transport=bar1" invalidates every number taken afterwards --
    which is exactly what would have happened in prefill.
    """

    def _comm(self, transport, group="tp:0"):
        from sglang.srt.distributed.device_communicators.barlink import (
            BarlinkCommunicator,
        )

        c = BarlinkCommunicator.__new__(BarlinkCommunicator)
        c.transport = transport
        c._path_dispatcher = None
        c._fallback_reported = set()
        c.group = group
        return c

    def _select(self, c, op, nbytes):
        from sglang.srt.distributed.device_communicators import barlink as mod

        with mock.patch.object(mod, "graph_capture_running", lambda: False):
            return mod.BarlinkCommunicator._select(c, op, nbytes)

    def test_an_uncovered_size_is_announced_with_op_bytes_and_reason(self):
        c = self._comm(_stub())
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.barlink", level="WARNING"
        ) as protokoll:
            self.assertIsNone(self._select(c, "reduce_scatter", 65536))
        text = "\n".join(protokoll.output)
        self.assertIn("reduce_scatter", text)
        self.assertIn("65536", text)
        self.assertIn("tp:0", text)
        self.assertIn("falling back", text)
        self.assertIn("Reason:", text)

    def test_a_covered_size_says_nothing(self):
        """The negative control. A notice that always fires is not one."""
        c = self._comm(_stub())
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.barlink"
        )
        with mock.patch.object(logger, "warning") as warnung:
            self.assertIsNotNone(self._select(c, "all_reduce", _bytes(2048)))
            self.assertIsNotNone(self._select(c, "all_reduce", _bytes(8192)))
            self.assertIsNotNone(self._select(c, "broadcast", 128))
        warnung.assert_not_called()

    def test_the_size_that_used_to_fall_through_silently_is_now_covered(self):
        """The original trigger case: 4096-token prefill. No notice,
        because no fallback -- that is the whole point of the rounds."""
        c = self._comm(_stub())
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.barlink"
        )
        with mock.patch.object(logger, "warning") as warnung:
            self.assertIsNotNone(self._select(c, "all_reduce", _bytes(4096)))
        warnung.assert_not_called()

    def test_it_speaks_once_per_op_and_size_class(self):
        """In the hot path the same sizes recur a thousandfold."""
        c = self._comm(_stub())
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.barlink"
        )
        with mock.patch.object(logger, "warning") as warnung:
            for _ in range(50):
                self._select(c, "reduce_scatter", 65536)
            self.assertEqual(warnung.call_count, 1)
            # A different size class is a new operating point.
            self._select(c, "reduce_scatter", 65536 * 64)
            self.assertEqual(warnung.call_count, 2)

    def test_each_group_speaks_for_itself(self):
        """tp and dcp get differently sized windows -- what fits in one
        need not fit in the other."""
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.barlink"
        )
        with mock.patch.object(logger, "warning") as warnung:
            for group in ("tp:0", "dcp:0"):
                self._select(self._comm(_stub(), group), "reduce_scatter", 65536)
            self.assertEqual(warnung.call_count, 2)

    def test_under_capture_the_bar_still_wins(self):
        """Under capture there is no fallback, hence no notice about one
        either -- it aborts there instead, and that stays that way."""
        from sglang.srt.distributed.device_communicators import barlink as mod

        c = self._comm(_stub())
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            with self.assertRaises(RuntimeError):
                mod.BarlinkCommunicator._select(c, "reduce_scatter", 65536)


class TestWhyNot(CustomTestCase):
    """The reason text. Diagnostic only -- it decides nothing."""

    def test_a_covered_size_has_no_reason(self):
        t = _stub()
        self.assertEqual(t.why_not("all_reduce", _bytes(2048)), "")

    def test_an_unaligned_payload_says_so(self):
        t = _stub()
        self.assertIn("multiple of 16", t.why_not("all_reduce", 4097))

    def test_too_many_rounds_names_the_cap(self):
        t = _stub(ar_max_rounds=2)
        reason = t.why_not("all_reduce", _bytes(8192))
        self.assertIn("rounds", reason)
        self.assertIn("2 are allowed", reason)

    def test_an_op_outside_the_coverage_says_so(self):
        self.assertIn("BARLINK_OPS", _stub().why_not("reduce_scatter", 4096))

    def test_a_transport_that_is_not_up_says_so(self):
        t = _stub(_up=False)
        self.assertIn("not set up", t.why_not("all_reduce", 4096))

    def test_a_switched_off_op_says_which_switch(self):
        t = _stub(bc_on=False)
        self.assertIn("SGLANG_BARLINK_BAR1_BC", t.why_not("broadcast", 128))


if __name__ == "__main__":
    unittest.main()
