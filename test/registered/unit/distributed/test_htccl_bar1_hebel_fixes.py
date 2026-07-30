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
   ``SGLANG_HTCCL_BAR1_GRAPH_GRID=1`` recovered 1337.2). The default now
   comes from ``SGLANG_HTCCL_GRAPH_ENABLE`` -- same question, same gate.

2. **The pipe range** took a quarter away from the all_reduce slot
   (8188 -> 6140 KiB, tipping point 2456 -> 1842 tokens, i.e. below the
   2048 working point). The measurement report attributed this to the
   result ring; that was wrong, and the arithmetic here shows both: the
   ring was not even present in that arm (``PIPE_DIRECT=0``), and the
   6140 falls out exactly from the pipe's extra slot set.

3. **The result ring** broke off the graph-safe direct mode during
   capture WARMUP, and ``SGLANG_HTCCL_BAR1_PIPE_RESULT_RING`` did not
   help, because the eager count was a constant.
"""

import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.htccl_bar1 import (
    HTCCLBar1Transport,
    geometry,
    graph_grid_default,
    max_payload,
)
from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
    ERG_EAGER_PLAETZE,
    result_slot_split,
    result_eager_free_slot,
    result_eager_slot,
    erg_eager_slack,
    result_graph_slot,
    pipe_range_bytes,
    pipe_window_requirement,
    pipe_plan,
    pipe_slot_default,
)
from sglang.srt.distributed.parallel_state import graph_freigabe_gesetzt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The geometry of the measurement run: 96 MiB window per rank, three
#: ranks, a2a on, pipe depth T=4, chunk target 1 MiB. Every number in this
#: test module falls out of these five.
FENSTER = 96 << 20
WELT = 3
TIEFE = 4
CHUNK_ZIEL = 1 << 20
K_MAX = 64
#: Hidden width x element size of the model -- so a tipping point can be
#: read in TOKENS, the way the measurement report writes it.
BYTE_JE_TOKEN = 5120 * 2
#: The working point of the run.
ARBEITSPUNKT_TOKEN = 2048


def _kipp_token(max_bytes: int) -> int:
    """Largest batch that ONE round still carries."""
    return max_bytes // BYTE_JE_TOKEN


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

    def test_without_the_release_the_reservation_stands(self):
        self.assertFalse(graph_grid_default({}))
        self.assertFalse(
            graph_grid_default({"SGLANG_HTCCL_GRAPH_ENABLE": "0"})
        )

    def test_the_release_carries_the_default(self):
        self.assertTrue(
            graph_grid_default({"SGLANG_HTCCL_GRAPH_ENABLE": "1"})
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
            graph_grid_default({"SGLANG_HTCCL_BAR1_GRAPH_GRID": "1"})
        )
        self.assertFalse(
            graph_grid_default({
                "SGLANG_HTCCL_GRAPH_ENABLE": "1",
                "SGLANG_HTCCL_BAR1_GRAPH_GRID": "0",
            })
        )
        self.assertTrue(
            graph_grid_default({
                "SGLANG_HTCCL_GRAPH_ENABLE": "0",
                "SGLANG_HTCCL_BAR1_GRAPH_GRID": "1",
            })
        )

    def test_the_two_readers_of_the_release_agree_word_for_word(self):
        """The same variable, read in two places -- so the same verdict.

        ``parallel_state.graph_freigabe_gesetzt`` decides whether bar1 may
        be captured at all; the default here decides HOW. If the two read
        an ``"aus"`` or an empty field differently, the difference only
        shows up later, as a throughput loss.
        """
        import os

        for wert in ("", "0", "1", "nein", "aus", "false", "ja", "wahr", "2"):
            with mock.patch.dict(
                os.environ, {"SGLANG_HTCCL_GRAPH_ENABLE": wert}, clear=False
            ):
                self.assertEqual(
                    graph_grid_default(),
                    graph_freigabe_gesetzt(),
                    f"value {wert!r} is read differently",
                )

    def test_an_unset_variable_is_not_the_same_as_a_zero(self):
        """The distinction the override hinges on.

        ``os.environ.get(name)`` returns ``None`` for "not set at all" and
        ``""`` for "set empty". Only the former is allowed to reach
        through to the release -- otherwise any empty assignment would be
        a silent enable.
        """
        self.assertTrue(
            graph_grid_default({"SGLANG_HTCCL_GRAPH_ENABLE": "1"})
        )
        self.assertFalse(
            graph_grid_default({
                "SGLANG_HTCCL_GRAPH_ENABLE": "1",
                "SGLANG_HTCCL_BAR1_GRAPH_GRID": "",
            })
        )


class TestKernelChoosesByTheDefault(CustomTestCase):
    """Und die Vorgabe kommt auch wirklich bis in ``_kernel``."""

    def _stub(self, graph_gitter: bool):
        t = HTCCLBar1Transport.__new__(HTCCLBar1Transport)
        t.graph_gitter = graph_gitter
        t._graph_gitter_gemeldet = False
        return t

    def _mit_erfassung(self):
        return mock.patch(
            "sglang.srt.distributed.device_communicators.htccl."
            "graph_capture_running",
            lambda: True,
        )

    def test_below_the_threshold_nothing_changes(self):
        for grid in (False, True):
            t = self._stub(grid)
            with self._mit_erfassung():
                self.assertEqual(t._kernel(1024, 4 << 20, "all_reduce"), 0)

    def test_capture_with_the_release_keeps_the_cooperative_launch(self):
        t = self._stub(True)
        with self._mit_erfassung():
            self.assertEqual(t._kernel(8 << 20, 4 << 20, "all_reduce"), 1)

    def test_capture_without_it_still_falls_back_and_says_so(self):
        t = self._stub(False)
        with self._mit_erfassung():
            self.assertEqual(t._kernel(8 << 20, 4 << 20, "all_reduce"), 0)
        self.assertTrue(t._graph_gitter_gemeldet)


# ===========================================================================
# Fix 2: who really takes the slot away from the window
# ===========================================================================


class TestWhoStealsTheSlot(CustomTestCase):
    """The measurement report's attribution, recomputed instead of taken on faith.

    The report attributed the loss to the result ring. But the pipe arm
    ran with ``SGLANG_HTCCL_BAR1_PIPE_DIRECT=0``, which makes the ring
    zero (``htccl_bar1.py``, "if not self.pipe_an or not self.pipe_direkt").
    Exactly one cause remains, and it matches the report's numbers to the
    byte.
    """

    def test_the_measured_slot_without_the_pipe_is_reproduced(self):
        n = max_payload(WELT, FENSTER, True, False, 0)
        geo = geometry(WELT, n, True, False, 0)
        self.assertEqual(geo["chunk_max"] // 1024, 8188)
        self.assertEqual(_kipp_token(n), 2456)

    def test_the_measured_loss_comes_from_the_pipe_set_not_from_the_ring(self):
        """6140 KiB falls out at ``erg_ring = 0`` -- the ring was not involved."""
        n = max_payload(WELT, FENSTER, True, True, 0)
        geo = geometry(WELT, n, True, True, 0)
        self.assertEqual(geo["chunk_max"] // 1024, 6140)
        self.assertEqual(_kipp_token(n), 1842)
        self.assertLess(
            _kipp_token(n), ARBEITSPUNKT_TOKEN,
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
        geo = geometry(WELT, 8 << 20, True, True, 0)
        self.assertEqual(geo["erg_ring"], 0)
        self.assertEqual(geo["off_erg"], -1)
        self.assertEqual(geo["erg_stride"], 0)
        # And with direct mode it costs what it costs -- that is not a
        # bug, it is the price of the mode.
        mit = geometry(WELT, 8 << 20, True, True, 2)
        self.assertGreater(mit["region_bytes"], geo["region_bytes"])

    def test_the_pipe_area_is_exactly_the_difference(self):
        """The difference between the two denominators, without a detour through the fixed points.

        12 slots against 16: mesh 4, ring 4, a2a 4, pipe 4 (2(R-1) each).
        Both of the measurement report's numbers fall out of the same
        division.
        """
        seite = 4096
        for nenner, erwartet in ((12, 8188), (16, 6140)):
            self.assertEqual(
                ((FENSTER - seite) // nenner // seite) * seite // 1024,
                erwartet,
            )
        self.assertEqual(
            geometry(WELT, max_payload(WELT, FENSTER, True, False, 0),
                      True, False, 0)["chunk_max"],
            ((FENSTER - seite) // 12 // seite) * seite,
        )
        self.assertEqual(
            geometry(WELT, max_payload(WELT, FENSTER, True, True, 0),
                      True, True, 0)["chunk_max"],
            ((FENSTER - seite) // 16 // seite) * seite,
        )


class TestPipeRangeByNeed(CustomTestCase):
    """The range the pipe really needs -- and what it hands back."""

    def _bereich(self) -> int:
        return pipe_range_bytes(
            WELT, TIEFE, pipe_slot_default(WELT, CHUNK_ZIEL)
        )

    def test_the_need_is_a_property_of_the_chunk_target_not_of_the_window(self):
        """The decoupling the whole computation depends on.

        The requirement depends on ``pipe_chunk_bytes``, ``T`` and ``R`` --
        on nothing that follows from the window. That is why it is a
        constant, not another denominator, in ``max_payload``'s
        fixed-point computation.
        """
        for window in (64 << 20, 96 << 20, 256 << 20, 8 << 30):
            n = max_payload(WELT, window, True, True, 0, self._bereich())
            geo = geometry(WELT, n, True, True, 0, self._bereich())
            self.assertEqual(geo["pipe_bereich"], self._bereich())

    def test_the_right_sized_area_lifts_the_tipping_point_over_the_working_point(self):
        n = max_payload(WELT, FENSTER, True, True, 0, self._bereich())
        geo = geometry(WELT, n, True, True, 0, self._bereich())
        self.assertEqual(geo["chunk_max"] // 1024, 7736)
        self.assertEqual(_kipp_token(n), 2320)
        self.assertGreater(_kipp_token(n), ARBEITSPUNKT_TOKEN)

    def test_pure_pipe_without_direct_keeps_almost_the_whole_slot(self):
        """"Pure pipe keeps the full slot", as far as that goes.

        It never gets fully back: the pipe genuinely needs its 5.3 MiB,
        and that is missing from the slot. Of the 2048 KiB the full slot
        set cost, 1596 come back -- 78%.
        """
        ohne = geometry(
            WELT, max_payload(WELT, FENSTER, True, False, 0), True, False, 0
        )["chunk_max"]
        alt = geometry(
            WELT, max_payload(WELT, FENSTER, True, True, 0), True, True, 0
        )["chunk_max"]
        neu = geometry(
            WELT, max_payload(WELT, FENSTER, True, True, 0, self._bereich()),
            True, True, 0, self._bereich(),
        )["chunk_max"]
        self.assertLess(alt, neu)
        self.assertLess(neu, ohne)
        self.assertGreater((neu - alt) / (ohne - alt), 0.75)

    def test_the_eager_direct_slots_are_charged_to_the_direct_mode(self):
        """Two slots stay two slots -- they belong to direct mode.

        The eager path of direct mode genuinely needs its slots; deducting
        them from the pipe range would mean showing direct mode as free.
        What it costs, it costs.
        """
        ber = self._bereich()
        ohne_ring = max_payload(WELT, FENSTER, True, True, 0, ber)
        mit_ring = max_payload(WELT, FENSTER, True, True, 2, ber)
        self.assertLess(mit_ring, ohne_ring)

    def test_every_payload_the_window_carries_finds_a_chunk_count(self):
        """A tight slot is allowed to cost coverage -- here it costs none.

        ``pipe_plan`` searches its K upward; if no K fits, the path
        withdraws via ``handles()``. That would be correct, but expensive.
        So this checks that, between the pipe's lower bound and the
        window's largest payload, it never actually comes to that.
        """
        slot = pipe_slot_default(WELT, CHUNK_ZIEL)
        maxb = max_payload(WELT, FENSTER, True, True, 0, self._bereich())
        schritt = 16 * 997
        nb = 256 << 10
        geprueft = 0
        while nb <= maxb:
            self.assertIsNotNone(
                pipe_plan(nb, WELT, slot, TIEFE, 0, CHUNK_ZIEL, K_MAX),
                f"{nb} bytes are not carried by the {slot}-byte slot",
            )
            geprueft += 1
            nb += schritt
        self.assertGreater(geprueft, 1000)

    def test_what_the_kernel_touches_stays_inside_what_the_layout_reserves(self):
        """The one condition a too-small range would violate.

        The kernel runs two rings of ``T(R-1)`` slots each; the highest
        address is ``2 T (R-1) * slot`` from ``off_pipe``. The reserved
        range is that same number, rounded up to a page -- and the result
        ring only starts beyond it.
        """
        for welt in (2, 3, 4, 8):
            for tiefe in (2, 4, 8):
                for ziel in (256 << 10, 1 << 20, 4 << 20):
                    slot = pipe_slot_default(welt, ziel)
                    ber = pipe_range_bytes(welt, tiefe, slot)
                    self.assertGreaterEqual(
                        ber, pipe_window_requirement(welt, tiefe, slot)
                    )
                    self.assertEqual(ber % 4096, 0)
                    self.assertEqual(slot % 16, 0)
                    n = max_payload(welt, FENSTER, True, True, 2, ber)
                    if n <= 0:
                        continue
                    geo = geometry(welt, n, True, True, 2, ber)
                    self.assertEqual(geo["pipe_bereich"], ber)
                    self.assertEqual(
                        geo["off_erg"], geo["off_pipe"] + ber
                    )
                    self.assertLessEqual(
                        geo["off_erg"] + 2 * geo["erg_stride"],
                        geo["region_bytes"],
                    )
                    self.assertEqual(geo["off_pipe"] % 4096, 0)
                    self.assertEqual(geo["off_erg"] % 4096, 0)

    def test_a_layout_without_the_area_is_caught_by_the_same_check(self):
        """The falsifier: the check above must have teeth.

        The plausible bug is a result ring appended right behind
        ``off_pipe`` because someone forgot the range. It would then land
        in the middle of the pipe slots.
        """
        ber = self._bereich()
        n = max_payload(WELT, FENSTER, True, True, 2, ber)
        geo = geometry(WELT, n, True, True, 2, ber)
        falsch = geo["off_pipe"]
        self.assertLess(
            falsch,
            geo["off_pipe"] + pipe_window_requirement(
                WELT, TIEFE, pipe_slot_default(WELT, CHUNK_ZIEL)
            ),
        )
        self.assertNotEqual(falsch, geo["off_erg"])

    def test_the_old_cut_is_still_available_and_byte_identical(self):
        """``pipe_bereich = 0`` means "as before" -- exactly that."""
        for max_bytes in (64 << 10, 8 << 20):
            alt = geometry(WELT, max_bytes, True, True, 2, 0)
            self.assertEqual(
                alt["pipe_bereich"],
                2 * (WELT - 1) * alt["chunk_max"],
            )
            self.assertEqual(
                alt["off_erg"], alt["off_pipe"] + alt["pipe_bereich"]
            )


# ===========================================================================
# Fix 3: the eager part of the result ring
# ===========================================================================


class TestResultSlotSplitIsConfigurable(CustomTestCase):
    def test_the_default_is_unchanged(self):
        self.assertEqual(ERG_EAGER_PLAETZE, 2)
        self.assertEqual(result_slot_split(5, True), (2, 3))
        self.assertEqual(result_slot_split(5, False), (5, 0))

    def test_a_bigger_ring_used_to_hand_out_graph_slots_only(self):
        """The measurement run's finding, as a number.

        ``ERG_RING=5`` gave three GRAPH slots and left the eager count at
        two -- and the abort happened during capture WARMUP, which runs
        eager. So the knob could not have worked, and ``ERG_RING=50``
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
        voriger = -1
        folge = []
        for _ in range(6):
            voriger = result_eager_free_slot(voriger, 3, [False] * 3)
            folge.append(voriger)
        self.assertEqual(folge, [0, 1, 2, 0, 1, 2])
        self.assertEqual(
            folge[:3],
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
        zuletzt = [None] * L
        zaehler = 0
        gesehen = []
        for _ in range(3 * L):
            i = result_eager_free_slot(
                (zaehler - 1) % L if zaehler else -1, L, [False] * L
            )
            gesehen.append(erg_eager_slack(i, zaehler, zuletzt, L))
            zuletzt[i] = zaehler
            zaehler += 1
        self.assertEqual(gesehen[:L], [L] * L)
        self.assertEqual(gesehen[L:], [L] * (2 * L))

    def test_a_skipped_slot_lowers_the_slack(self):
        L = 3
        zuletzt = [None, None, None]
        # Slot 0 at call 0, then slot 0 again at call 1.
        zuletzt[0] = 0
        self.assertEqual(erg_eager_slack(0, 1, zuletzt, L), 1)
        self.assertEqual(erg_eager_slack(0, 2, zuletzt, L), 2)
        self.assertEqual(erg_eager_slack(0, 9, zuletzt, L), L)

    def test_an_unused_slot_may_take_the_full_distance(self):
        self.assertEqual(erg_eager_slack(1, 7, [None, None], 2), 2)

    def test_the_slack_is_never_zero(self):
        """``0`` would switch off the handshake in the kernel entirely."""
        for zaehler in range(5):
            self.assertGreaterEqual(
                erg_eager_slack(0, zaehler, [zaehler], 4), 1
            )


def _stub(**kw):
    """A transport without ``__init__`` -- just the fields ``_result_slot`` needs."""
    t = HTCCLBar1Transport.__new__(HTCCLBar1Transport)
    t.pipe_direkt = True
    t.pipe_direkt_graph = False
    t._direkt_graph_gemeldet = False
    t._erg_graph_leer_gemeldet = False
    t._erg_graph_vergeben = 0
    t._erg_i = -1
    t._erg_lebt = [None, None]
    t._erg_zuletzt = [None, None]
    t._erg_zaehler = 0
    t._erg_eager_voll = 0
    t._erg_eager_voll_gemeldet = False
    t._erg_eager_plaetze = 2
    t._erg_graph_plaetze = 0
    t._eigen = (1 << 30, 0, 0)
    t._geo = {"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 2}
    t._pipe_ext = mock.Mock()
    t._pipe_ext.bar1_erg_tensor.side_effect = lambda ptr, muster: mock.Mock(
        name=f"erg@{ptr}"
    )
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _ohne_erfassung():
    return mock.patch(
        "sglang.srt.distributed.device_communicators.htccl."
        "graph_capture_running",
        lambda: False,
    )


class TestEagerFullFallsBackInsteadOfAborting(CustomTestCase):
    """The boot-time stopper of direct mode, at the seam.

    Previously ``_result_slot`` raised ``Bar1Unavailable`` here -- during
    capture WARMUP that meant a dead server. Aborting was the wrong answer
    to the right concern: what stays forbidden is writing into a held
    buffer, and that is precisely what does not happen at ``direkt=0``.
    It takes the same path the exhausted graph pool already takes a
    couple of lines above.
    """

    def test_a_held_slot_no_longer_kills_the_call(self):
        t = _stub()
        with _ohne_erfassung():
            behalten, _platz, _slack = t._result_slot(object())
            zweiter = t._result_slot(object())
        self.assertIsNotNone(behalten)
        self.assertIsNotNone(zweiter)
        self.assertNotEqual(zweiter[1], 0)

    def test_all_slots_held_falls_back_to_direct_zero_and_counts_it(self):
        t = _stub()
        gehalten = []
        with _ohne_erfassung():
            for _ in range(2):
                gehalten.append(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
        self.assertEqual(t._erg_eager_voll, 2)
        self.assertTrue(t._erg_eager_voll_gemeldet)
        # And as soon as the caller lets go, direct mode runs again.
        gehalten.clear()
        with _ohne_erfassung():
            self.assertIsNotNone(t._result_slot(object()))

    def test_more_eager_slots_carry_more_live_results(self):
        """The knob missing from the measurement run, checked at the seam."""
        t = _stub(
            _erg_lebt=[None] * 4,
            _erg_zuletzt=[None] * 4,
            _erg_eager_plaetze=4,
            _geo={"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 4},
        )
        gehalten = []
        with _ohne_erfassung():
            for _ in range(4):
                gehalten.append(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
        self.assertEqual([g[1] for g in gehalten], [0, 1, 2, 3])
        self.assertEqual(t._erg_eager_voll, 1)

    def test_the_handshake_slack_follows_the_real_reuse_distance(self):
        t = _stub(pipe_direkt_graph=True)
        with _ohne_erfassung():
            erster = t._result_slot(object())      # slot 0, gets HELD
            t._result_slot(object())               # slot 1, discarded right away
            dritter = t._result_slot(object())     # slot 1 again
        self.assertEqual(erster[1], 0)
        self.assertEqual(dritter[1], 1)
        # Slot 1 was one call back, not two.
        self.assertEqual(dritter[2], 1)

    def test_the_measured_rotation_keeps_its_old_slack(self):
        t = _stub(pipe_direkt_graph=True)
        with _ohne_erfassung():
            for _ in range(5):
                _out, _platz, slack = t._result_slot(object())
                self.assertEqual(slack, t._erg_eager_plaetze)


if __name__ == "__main__":
    unittest.main()
