"""The three fixes from the #293 leverage measurement, step 2.

The measurement run (``docs/dev/INTEGRATION_R3_VALIDATION.md``, section
"#293 step 2: leverage measurement against the prefill ceiling") named and
quantified three items. This test module pins down the answers by
computing them rather than arguing about them -- everything here is pure
arithmetic and state bookkeeping, no card involved.

1. **The grid reservation** pinned the cooperative launch to the slower
   ``1blk`` variant during every CUDA graph capture. Under the prefill
   graph that cost 16.1% throughput at eight sessions
   (1334.5 -> 1151.6 tok/s; the falsifier run with
   ``SGLANG_BARLINK_BAR1_GRAPH_GRID=1`` recovered 1337.2). The default now
   comes from ``SGLANG_BARLINK_GRAPH_ENABLE`` -- same question, same gate.

2. **The pipe range** took a quarter away from the all_reduce slot
   (8188 -> 6140 KiB, tipping point 2456 -> 1842 tokens, i.e. below the
   2048 working point). The measurement report attributed this to the
   result ring; that was wrong, and the arithmetic here shows both: the
   ring was not even present in that arm (``PIPE_DIRECT=0``), and the
   6140 falls out exactly from the pipe's extra slot set.

3. **The result ring** broke off the graph-safe direct mode during
   capture WARMUP, and ``SGLANG_BARLINK_BAR1_PIPE_RESULT_RING`` did not
   help, because the eager count was a constant.
"""

import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
    geometry,
    graph_grid_default,
    max_payload,
)
from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
    RESULT_EAGER_SLOTS,
    result_slot_split,
    result_eager_free_slot,
    result_eager_slot,
    result_eager_slack,
    result_graph_slot,
    pipe_range_bytes,
    pipe_window_requirement,
    pipe_plan,
    pipe_slot_default,
)
from sglang.srt.distributed.parallel_state import graph_enable_set
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The geometry of the measurement run: 96 MiB window per rank, three
#: ranks, a2a on, pipe depth T=4, chunk target 1 MiB. Every number in this
#: test module falls out of these five.
WINDOW = 96 << 20
WORLD = 3
DEPTH = 4
CHUNK_TARGET = 1 << 20
K_MAX = 64
#: Hidden width x element size of the model -- so a tipping point can be
#: read in TOKENS, the way the measurement report writes it.
BYTES_PER_TOKEN = 5120 * 2
#: The working point of the run.
WORKING_POINT_TOKENS = 2048


def _tipping_tokens(max_bytes: int) -> int:
    """Largest batch that ONE round still carries."""
    return max_bytes // BYTES_PER_TOKEN


# ===========================================================================
# Fix 1: the grid reservation depends on the graph release
# ===========================================================================


class TestGridDefaultFollowsTheRelease(CustomTestCase):
    """One gate, one answer -- not two switches for the same question.

    ``bar1_graph_check.py`` answers, with its ``grid`` case, exactly the
    question of whether ``cudaLaunchCooperativeKernel`` can be captured
    here. As long as a separate opt-in stood next to it, the gate could
    pass, the release could stand -- and the kernel would still fall back
    to ``1blk``. That is exactly what happened, and it cost 16.1%.
    """

    def test_unset_release_follows_the_canonical_default(self):
        # Since the #369 release (8d9a9ec314) graph_enable_set() defaults ON
        # when the env var is unset; graph_grid_default() must mirror that
        # default instead of silently keeping the pre-release reservation.
        # The explicit off-ramp still stands.
        self.assertTrue(graph_grid_default({}))
        self.assertFalse(
            graph_grid_default({"SGLANG_BARLINK_GRAPH_ENABLE": "0"})
        )

    def test_the_release_carries_the_default(self):
        self.assertTrue(
            graph_grid_default({"SGLANG_BARLINK_GRAPH_ENABLE": "1"})
        )

    def test_the_override_wins_in_both_directions(self):
        """Both directions, because both are needed.

        The ``grid`` gate case runs the cooperative launch WITHOUT the
        release (it is supposed to be the thing that justifies it in the
        first place); the ``vorbehalt`` gate case runs the fallback WITH
        the release standing. A switch that only works in one direction
        would make one of the two cases untestable.
        """
        self.assertTrue(
            graph_grid_default({"SGLANG_BARLINK_BAR1_GRAPH_GRID": "1"})
        )
        self.assertFalse(
            graph_grid_default({
                "SGLANG_BARLINK_GRAPH_ENABLE": "1",
                "SGLANG_BARLINK_BAR1_GRAPH_GRID": "0",
            })
        )
        self.assertTrue(
            graph_grid_default({
                "SGLANG_BARLINK_GRAPH_ENABLE": "0",
                "SGLANG_BARLINK_BAR1_GRAPH_GRID": "1",
            })
        )

    def test_the_two_readers_of_the_release_agree_word_for_word(self):
        """The same variable, read in two places -- so the same verdict.

        ``parallel_state.graph_enable_set`` decides whether bar1 may
        be captured at all; the default here decides HOW. If the two read
        an ``"off"`` or an empty field differently, the difference only
        shows up later, as a throughput loss.
        """
        import os

        for value in ("", "0", "1", "no", "off", "false", "yes", "true", "2"):
            with mock.patch.dict(
                os.environ, {"SGLANG_BARLINK_GRAPH_ENABLE": value}, clear=False
            ):
                self.assertEqual(
                    graph_grid_default(),
                    graph_enable_set(),
                    f"value {value!r} is read differently",
                )

    def test_an_unset_variable_is_not_the_same_as_a_zero(self):
        """The distinction the override hinges on.

        ``os.environ.get(name)`` returns ``None`` for "not set at all" and
        ``""`` for "set empty". Only the former is allowed to reach
        through to the release -- otherwise any empty assignment would be
        a silent enable.
        """
        self.assertTrue(
            graph_grid_default({"SGLANG_BARLINK_GRAPH_ENABLE": "1"})
        )
        self.assertFalse(
            graph_grid_default({
                "SGLANG_BARLINK_GRAPH_ENABLE": "1",
                "SGLANG_BARLINK_BAR1_GRAPH_GRID": "",
            })
        )


class TestKernelChoosesByTheDefault(CustomTestCase):
    """And the default really does reach ``_kernel``."""

    def _stub(self, graph_grid: bool):
        t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
        t.graph_grid = graph_grid
        t._graph_grid_reported = False
        return t

    def _with_capture(self):
        return mock.patch(
            "sglang.srt.distributed.device_communicators.barlink."
            "graph_capture_running",
            lambda: True,
        )

    def test_below_the_threshold_nothing_changes(self):
        for grid in (False, True):
            t = self._stub(grid)
            with self._with_capture():
                self.assertEqual(t._kernel(1024, 4 << 20, "all_reduce"), 0)

    def test_capture_with_the_release_keeps_the_cooperative_launch(self):
        t = self._stub(True)
        with self._with_capture():
            self.assertEqual(t._kernel(8 << 20, 4 << 20, "all_reduce"), 1)

    def test_capture_without_it_still_falls_back_and_says_so(self):
        t = self._stub(False)
        with self._with_capture():
            self.assertEqual(t._kernel(8 << 20, 4 << 20, "all_reduce"), 0)
        self.assertTrue(t._graph_grid_reported)


# ===========================================================================
# Fix 2: who really takes the slot away from the window
# ===========================================================================


class TestWhoStealsTheSlot(CustomTestCase):
    """The measurement report's attribution, recomputed instead of taken on faith.

    The report attributed the loss to the result ring. But the pipe arm
    ran with ``SGLANG_BARLINK_BAR1_PIPE_DIRECT=0``, which makes the ring
    zero (``barlink_bar1.py``, "if not self.pipe_on or not self.pipe_direct").
    Exactly one cause remains, and it matches the report's numbers to the
    byte.
    """

    def test_the_measured_slot_without_the_pipe_is_reproduced(self):
        n = max_payload(WORLD, WINDOW, True, False, 0)
        geo = geometry(WORLD, n, True, False, 0)
        self.assertEqual(geo["chunk_max"] // 1024, 8188)
        self.assertEqual(_tipping_tokens(n), 2456)

    def test_the_measured_loss_comes_from_the_pipe_set_not_from_the_ring(self):
        """6140 KiB falls out at ``result_ring = 0`` -- the ring was not involved."""
        n = max_payload(WORLD, WINDOW, True, True, 0)
        geo = geometry(WORLD, n, True, True, 0)
        self.assertEqual(geo["chunk_max"] // 1024, 6140)
        self.assertEqual(_tipping_tokens(n), 1842)
        self.assertLess(
            _tipping_tokens(n), WORKING_POINT_TOKENS,
            "the tipping point must lie BELOW the working point -- "
            "otherwise it does not explain the two rounds of the "
            "20 MiB all_reduce",
        )

    def test_the_ring_is_not_reserved_without_the_direct_mode(self):
        """The claim "the ring always takes its share", checked.

        Without direct mode there is no result ring in the geometry: no
        offset, no stride, no length. The regression guard for this,
        because a ring reserved without being used would be exactly the
        bug the measurement report suspected.
        """
        geo = geometry(WORLD, 8 << 20, True, True, 0)
        self.assertEqual(geo["result_ring"], 0)
        self.assertEqual(geo["off_result"], -1)
        self.assertEqual(geo["result_stride"], 0)
        # And with direct mode it costs what it costs -- that is not a
        # bug, it is the price of the mode.
        with_ring = geometry(WORLD, 8 << 20, True, True, 2)
        self.assertGreater(with_ring["region_bytes"], geo["region_bytes"])

    def test_the_pipe_area_is_exactly_the_difference(self):
        """The difference between the two denominators, without a detour through the fixed points.

        12 slots against 16: mesh 4, ring 4, a2a 4, pipe 4 (2(R-1) each).
        Both of the measurement report's numbers fall out of the same
        division.
        """
        page = 4096
        for denominator, expected in ((12, 8188), (16, 6140)):
            self.assertEqual(
                ((WINDOW - page) // denominator // page) * page // 1024,
                expected,
            )
        self.assertEqual(
            geometry(WORLD, max_payload(WORLD, WINDOW, True, False, 0),
                      True, False, 0)["chunk_max"],
            ((WINDOW - page) // 12 // page) * page,
        )
        self.assertEqual(
            geometry(WORLD, max_payload(WORLD, WINDOW, True, True, 0),
                      True, True, 0)["chunk_max"],
            ((WINDOW - page) // 16 // page) * page,
        )


class TestPipeRangeByNeed(CustomTestCase):
    """The range the pipe really needs -- and what it hands back."""

    def _range_bytes(self) -> int:
        return pipe_range_bytes(
            WORLD, DEPTH, pipe_slot_default(WORLD, CHUNK_TARGET)
        )

    def test_the_need_is_a_property_of_the_chunk_target_not_of_the_window(self):
        """The decoupling the whole computation depends on.

        The requirement depends on ``pipe_chunk_bytes``, ``T`` and ``R`` --
        on nothing that follows from the window. That is why it is a
        constant, not another denominator, in ``max_payload``'s
        fixed-point computation.
        """
        for window in (64 << 20, 96 << 20, 256 << 20, 8 << 30):
            n = max_payload(WORLD, window, True, True, 0, self._range_bytes())
            geo = geometry(WORLD, n, True, True, 0, self._range_bytes())
            self.assertEqual(geo["pipe_range"], self._range_bytes())

    def test_the_right_sized_area_lifts_the_tipping_point_over_the_working_point(self):
        n = max_payload(WORLD, WINDOW, True, True, 0, self._range_bytes())
        geo = geometry(WORLD, n, True, True, 0, self._range_bytes())
        self.assertEqual(geo["chunk_max"] // 1024, 7736)
        self.assertEqual(_tipping_tokens(n), 2320)
        self.assertGreater(_tipping_tokens(n), WORKING_POINT_TOKENS)

    def test_pure_pipe_without_direct_keeps_almost_the_whole_slot(self):
        """"Pure pipe keeps the full slot", as far as that goes.

        It never gets fully back: the pipe genuinely needs its 5.3 MiB,
        and that is missing from the slot. Of the 2048 KiB the full slot
        set cost, 1596 come back -- 78%.
        """
        without = geometry(
            WORLD, max_payload(WORLD, WINDOW, True, False, 0), True, False, 0
        )["chunk_max"]
        old = geometry(
            WORLD, max_payload(WORLD, WINDOW, True, True, 0), True, True, 0
        )["chunk_max"]
        new_ = geometry(
            WORLD, max_payload(WORLD, WINDOW, True, True, 0, self._range_bytes()),
            True, True, 0, self._range_bytes(),
        )["chunk_max"]
        self.assertLess(old, new_)
        self.assertLess(new_, without)
        self.assertGreater((new_ - old) / (without - old), 0.75)

    def test_the_eager_direct_slots_are_charged_to_the_direct_mode(self):
        """Two slots stay two slots -- they belong to direct mode.

        The eager path of direct mode genuinely needs its slots; deducting
        them from the pipe range would mean showing direct mode as free.
        What it costs, it costs.
        """
        range_bytes = self._range_bytes()
        without_ring = max_payload(WORLD, WINDOW, True, True, 0, range_bytes)
        with_ring = max_payload(WORLD, WINDOW, True, True, 2, range_bytes)
        self.assertLess(with_ring, without_ring)

    def test_every_payload_the_window_carries_finds_a_chunk_count(self):
        """A tight slot is allowed to cost coverage -- here it costs none.

        ``pipe_plan`` searches its K upward; if no K fits, the path
        withdraws via ``handles()``. That would be correct, but expensive.
        So this checks that, between the pipe's lower bound and the
        window's largest payload, it never actually comes to that.
        """
        slot = pipe_slot_default(WORLD, CHUNK_TARGET)
        maxb = max_payload(WORLD, WINDOW, True, True, 0, self._range_bytes())
        step = 16 * 997
        nb = 256 << 10
        checked = 0
        while nb <= maxb:
            self.assertIsNotNone(
                pipe_plan(nb, WORLD, slot, DEPTH, 0, CHUNK_TARGET, K_MAX),
                f"{nb} bytes are not carried by the {slot}-byte slot",
            )
            checked += 1
            nb += step
        self.assertGreater(checked, 1000)

    def test_what_the_kernel_touches_stays_inside_what_the_layout_reserves(self):
        """The one condition a too-small range would violate.

        The kernel runs two rings of ``T(R-1)`` slots each; the highest
        address is ``2 T (R-1) * slot`` from ``off_pipe``. The reserved
        range is that same number, rounded up to a page -- and the result
        ring only starts beyond it.
        """
        for world in (2, 3, 4, 8):
            for depth in (2, 4, 8):
                for dst in (256 << 10, 1 << 20, 4 << 20):
                    slot = pipe_slot_default(world, dst)
                    range_bytes = pipe_range_bytes(world, depth, slot)
                    self.assertGreaterEqual(
                        range_bytes, pipe_window_requirement(world, depth, slot)
                    )
                    self.assertEqual(range_bytes % 4096, 0)
                    self.assertEqual(slot % 16, 0)
                    n = max_payload(world, WINDOW, True, True, 2, range_bytes)
                    if n <= 0:
                        continue
                    geo = geometry(world, n, True, True, 2, range_bytes)
                    self.assertEqual(geo["pipe_range"], range_bytes)
                    self.assertEqual(
                        geo["off_result"], geo["off_pipe"] + range_bytes
                    )
                    self.assertLessEqual(
                        geo["off_result"] + 2 * geo["result_stride"],
                        geo["region_bytes"],
                    )
                    self.assertEqual(geo["off_pipe"] % 4096, 0)
                    self.assertEqual(geo["off_result"] % 4096, 0)

    def test_a_layout_without_the_area_is_caught_by_the_same_check(self):
        """The falsifier: the check above must have teeth.

        The plausible bug is a result ring appended right behind
        ``off_pipe`` because someone forgot the range. It would then land
        in the middle of the pipe slots.
        """
        range_bytes = self._range_bytes()
        n = max_payload(WORLD, WINDOW, True, True, 2, range_bytes)
        geo = geometry(WORLD, n, True, True, 2, range_bytes)
        wrong = geo["off_pipe"]
        self.assertLess(
            wrong,
            geo["off_pipe"] + pipe_window_requirement(
                WORLD, DEPTH, pipe_slot_default(WORLD, CHUNK_TARGET)
            ),
        )
        self.assertNotEqual(wrong, geo["off_result"])

    def test_the_old_cut_is_still_available_and_byte_identical(self):
        """``pipe_range = 0`` means "as before" -- exactly that."""
        for max_bytes in (64 << 10, 8 << 20):
            old = geometry(WORLD, max_bytes, True, True, 2, 0)
            self.assertEqual(
                old["pipe_range"],
                2 * (WORLD - 1) * old["chunk_max"],
            )
            self.assertEqual(
                old["off_result"], old["off_pipe"] + old["pipe_range"]
            )


# ===========================================================================
# Fix 3: the eager part of the result ring
# ===========================================================================


class TestResultSlotSplitIsConfigurable(CustomTestCase):
    def test_the_default_is_unchanged(self):
        self.assertEqual(RESULT_EAGER_SLOTS, 2)
        self.assertEqual(result_slot_split(5, True), (2, 3))
        self.assertEqual(result_slot_split(5, False), (5, 0))

    def test_a_bigger_ring_used_to_hand_out_graph_slots_only(self):
        """The measurement run's finding, as a number.

        ``RESULT_RING=5`` gave three GRAPH slots and left the eager count at
        two -- and the abort happened during capture WARMUP, which runs
        eager. So the knob could not have worked, and ``RESULT_RING=50``
        would not have either.
        """
        for ring in (5, 8, 50):
            eager, _graph = result_slot_split(ring, True)
            self.assertEqual(eager, 2)

    def test_now_the_eager_share_is_the_parameter(self):
        self.assertEqual(result_slot_split(5, True, 4), (4, 1))
        self.assertEqual(result_slot_split(5, True, 5), (5, 0))
        self.assertEqual(result_slot_split(5, True, 9), (5, 0))
        self.assertEqual(result_slot_split(5, False, 4), (5, 0))

    def test_the_graph_supply_still_begins_behind_the_eager_share(self):
        eager, graph = result_slot_split(7, True, 4)
        self.assertEqual((eager, graph), (4, 3))
        self.assertEqual(
            [result_graph_slot(v, eager, graph) for v in range(4)],
            [4, 5, 6, None],
        )


class TestFreeEagerSlot(CustomTestCase):
    def test_with_everything_free_the_rotation_is_the_old_one(self):
        previous = -1
        sequence = []
        for _ in range(6):
            previous = result_eager_free_slot(previous, 3, [False] * 3)
            sequence.append(previous)
        self.assertEqual(sequence, [0, 1, 2, 0, 1, 2])
        self.assertEqual(
            sequence[:3],
            [result_eager_slot(-1, 3), result_eager_slot(0, 3),
             result_eager_slot(1, 3)],
        )

    def test_a_held_slot_is_skipped_instead_of_aborting(self):
        self.assertEqual(result_eager_free_slot(-1, 2, [True, False]), 1)
        self.assertEqual(result_eager_free_slot(0, 3, [False, True, False]), 2)

    def test_all_held_means_no_slot_and_that_is_an_answer(self):
        self.assertIsNone(result_eager_free_slot(-1, 2, [True, True]))
        self.assertIsNone(result_eager_free_slot(1, 4, [True] * 4))

    def test_a_ring_without_slots_is_a_programming_error_not_a_none(self):
        with self.assertRaises(ValueError):
            result_eager_free_slot(-1, 0, [])


class TestSlackIsALowerBound(CustomTestCase):
    """A slack that is too LARGE would be the weaker wait condition.

    The kernel waits for the peer to have entered generation
    ``TARGET - slack + 1``. The larger the slack, the earlier the
    condition fires -- the dangerous direction. It must therefore be
    allowed to underestimate the actual reuse distance, never
    overestimate it.
    """

    def test_strict_rotation_gives_exactly_the_number_of_slots(self):
        L = 3
        last_used = [None] * L
        counter = 0
        seen = []
        for _ in range(3 * L):
            i = result_eager_free_slot(
                (counter - 1) % L if counter else -1, L, [False] * L
            )
            seen.append(result_eager_slack(i, counter, last_used, L))
            last_used[i] = counter
            counter += 1
        self.assertEqual(seen[:L], [L] * L)
        self.assertEqual(seen[L:], [L] * (2 * L))

    def test_a_skipped_slot_lowers_the_slack(self):
        L = 3
        last_used = [None, None, None]
        # Slot 0 at call 0, then slot 0 again at call 1.
        last_used[0] = 0
        self.assertEqual(result_eager_slack(0, 1, last_used, L), 1)
        self.assertEqual(result_eager_slack(0, 2, last_used, L), 2)
        self.assertEqual(result_eager_slack(0, 9, last_used, L), L)

    def test_an_unused_slot_may_take_the_full_distance(self):
        self.assertEqual(result_eager_slack(1, 7, [None, None], 2), 2)

    def test_the_slack_is_never_zero(self):
        """``0`` would switch off the handshake in the kernel entirely."""
        for counter in range(5):
            self.assertGreaterEqual(
                result_eager_slack(0, counter, [counter], 4), 1
            )


def _stub(**kw):
    """A transport without ``__init__`` -- just the fields ``_result_slot`` needs."""
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.pipe_direct = True
    t.pipe_direct_graph = False
    t._direct_graph_reported = False
    t._result_graph_empty_reported = False
    t._result_graph_assigned = 0
    t._result_i = -1
    t._result_alive = [None, None]
    t._result_last = [None, None]
    t._result_counter = 0
    t._result_eager_full = 0
    t._result_eager_full_reported = False
    t._result_eager_slots = 2
    t._result_graph_slots = 0
    t._own = (1 << 30, 0, 0)
    t._geo = {"off_result": 4096, "result_stride": 1 << 20, "result_ring": 2}
    t._pipe_ext = mock.Mock()
    t._pipe_ext.bar1_result_tensor.side_effect = lambda ptr, like: mock.Mock(
        name=f"result@{ptr}"
    )
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _without_capture():
    return mock.patch(
        "sglang.srt.distributed.device_communicators.barlink."
        "graph_capture_running",
        lambda: False,
    )


class TestEagerFullFallsBackInsteadOfAborting(CustomTestCase):
    """The boot-time stopper of direct mode, at the seam.

    Previously ``_result_slot`` raised ``Bar1Unavailable`` here -- during
    capture WARMUP that meant a dead server. Aborting was the wrong answer
    to the right concern: what stays forbidden is writing into a held
    buffer, and that is precisely what does not happen at ``direct=0``.
    It takes the same path the exhausted graph pool already takes a
    couple of lines above.
    """

    def test_a_held_slot_no_longer_kills_the_call(self):
        t = _stub()
        with _without_capture():
            held, _slot, _slack = t._result_slot(object())
            second = t._result_slot(object())
        self.assertIsNotNone(held)
        self.assertIsNotNone(second)
        self.assertNotEqual(second[1], 0)

    def test_all_slots_held_falls_back_to_direct_zero_and_counts_it(self):
        t = _stub()
        held_list = []
        with _without_capture():
            for _ in range(2):
                held_list.append(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
        self.assertEqual(t._result_eager_full, 2)
        self.assertTrue(t._result_eager_full_reported)
        # And as soon as the caller lets go, direct mode runs again.
        held_list.clear()
        with _without_capture():
            self.assertIsNotNone(t._result_slot(object()))

    def test_more_eager_slots_carry_more_live_results(self):
        """The knob missing from the measurement run, checked at the seam."""
        t = _stub(
            _result_alive=[None] * 4,
            _result_last=[None] * 4,
            _result_eager_slots=4,
            _geo={"off_result": 4096, "result_stride": 1 << 20, "result_ring": 4},
        )
        held_list = []
        with _without_capture():
            for _ in range(4):
                held_list.append(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
        self.assertEqual([g[1] for g in held_list], [0, 1, 2, 3])
        self.assertEqual(t._result_eager_full, 1)

    def test_the_handshake_slack_follows_the_real_reuse_distance(self):
        t = _stub(pipe_direct_graph=True)
        with _without_capture():
            isFirst = t._result_slot(object())      # slot 0, gets HELD
            t._result_slot(object())               # slot 1, discarded right away
            third = t._result_slot(object())     # slot 1 again
        self.assertEqual(isFirst[1], 0)
        self.assertEqual(third[1], 1)
        # Slot 1 was one call back, not two.
        self.assertEqual(third[2], 1)

    def test_the_measured_rotation_keeps_its_old_slack(self):
        t = _stub(pipe_direct_graph=True)
        with _without_capture():
            for _ in range(5):
                _out, _slot, slack = t._result_slot(object())
                self.assertEqual(slack, t._result_eager_slots)


if __name__ == "__main__":
    unittest.main()
