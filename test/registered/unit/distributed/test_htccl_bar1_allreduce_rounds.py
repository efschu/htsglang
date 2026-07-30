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

from sglang.srt.distributed.device_communicators.htccl_bar1 import (
    HTCCLBar1Transport,
    a2a_runden,
    ar_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: Die Geometrie des Laufs, aus der Aufbau-Zeile: Region 96,0 MiB je Rang,
#: 12 Schlitze, Schlitz 8188 KiB, groesste Nutzlast 24564 KiB.
CHUNK_MAX = 8188 * 1024
SCHLITZ = CHUNK_MAX
WELT = 3
#: Verborgene Breite und Elementgroesse des Modells -- damit die Leiter in
#: TOKEN gelesen werden kann, so wie die Analyse sie aufschreibt.
HIDDEN, ELEM = 5120, 2
#: Der Kipp-Punkt, ausgerechnet: Nutzlast > 3 x 8188 KiB.
KIPP_TOKEN = 2456


def _bytes(token: int) -> int:
    return token * HIDDEN * ELEM


def _stub(rank=0, welt=WELT, **kw):
    t = HTCCLBar1Transport.__new__(HTCCLBar1Transport)
    t.welt = welt
    t.rank = rank
    t._auf = True
    t._ext = object()
    t._belege_stehen = True
    t._a2a_beleg = True
    t._bc_beleg = True
    t.a2a_an = True
    t.ag_an = True
    t.bc_an = True
    t.min_bytes = 4096
    t.max_bytes = 24564 * 1024
    t.a2a_min_bytes = 16
    t.ag_min_bytes = 1
    t.bc_min_bytes = 1
    t.ag_max_runden = 16
    t.bc_max_runden = 16
    t.ar_max_runden = 16
    t.a2a_max_runden = 16
    t.ring_ab = 1 << 20
    t.pipe_an = False
    t.pipe_ab = 256 << 10
    t._plan = None
    t._fenster_minimum = 96 << 20
    t._geo = {
        "off_a2a": 4096,
        "a2a_schlitz": SCHLITZ,
        "chunk_max": CHUNK_MAX,
        "region_bytes": 90 << 20,
    }
    for k, v in kw.items():
        if k in ("off_a2a", "a2a_schlitz", "chunk_max", "region_bytes"):
            t._geo[k] = v
        else:
            setattr(t, k, v)
    return t


class TestArPlanArithmetic(CustomTestCase):
    """The decomposition. Every byte once, every round inside the slot."""

    def _pruefe(self, nbytes, chunk_max=CHUNK_MAX, welt=WELT):
        plan = ar_plan(nbytes, chunk_max, welt)
        gesehen = []
        for versatz, laenge in plan:
            self.assertGreater(laenge, 0)
            self.assertEqual(laenge % 16, 0, msg="Runde ist kein Vielfaches von 16")
            self.assertEqual(versatz % 16, 0, msg="Versatz ist nicht ausgerichtet")
            # Der Wirt besteht auf einem 128-Bit-Paket JE RANG (TORCH_CHECK
            # n4 >= R). Eine Restrunde darunter waere kein langsamer Fall,
            # sondern ein Abbruch.
            self.assertGreaterEqual(
                laenge // 16, welt, msg=f"Runde {laenge} B unter einem Paket je Rang"
            )
            scherbe = -(-(laenge // 16) // welt) * 16
            self.assertLessEqual(scherbe, chunk_max, msg="Scherbe passt nicht")
            gesehen.extend(range(versatz, versatz + laenge))
        self.assertEqual(gesehen, list(range(nbytes)))
        return plan

    def test_the_working_point_stays_one_round(self):
        """2048 Token, 20,0 MiB -- der gemessene Arbeitspunkt.

        Er darf sich nicht aendern: dieselbe eine Runde wie vorher, sonst
        waeren alle Zahlen des Laufs auf einmal nicht mehr vergleichbar.
        """
        self.assertEqual(len(self._pruefe(_bytes(2048))), 1)

    def test_the_documented_tipping_point(self):
        """2456 Token passen noch, 2457 nicht mehr -- und laufen jetzt."""
        self.assertEqual(len(self._pruefe(_bytes(KIPP_TOKEN))), 1)
        self.assertEqual(len(self._pruefe(_bytes(KIPP_TOKEN + 1))), 2)

    def test_the_usual_chunked_prefill_sizes(self):
        """Die beiden Groessen, bei denen bar1 im Prefill still ausgesetzt
        haette."""
        self.assertEqual(len(self._pruefe(_bytes(4096))), 2)
        self.assertEqual(len(self._pruefe(_bytes(8192))), 4)

    def test_the_ladder_is_gapless(self):
        for token in (2048, 2456, 2457, 4096, 8192):
            self._pruefe(_bytes(token))

    def test_rounds_are_evenly_distributed(self):
        """Nicht bis zum Anschlag fuellen: der Schwanz waere sonst beliebig
        klein, und der Wirt lehnt eine Runde unter einem Paket je Rang ab."""
        for nbytes in (_bytes(2457), _bytes(4096) + 16, 25153536 + 16):
            plan = self._pruefe(nbytes)
            laengen = [laenge for _, laenge in plan]
            # Hoechstens ein Paket Unterschied zwischen groesster und
            # kleinster Runde.
            self.assertLessEqual(max(laengen) - min(laengen), 16, msg=str(nbytes))

    def test_a_tail_of_one_packet_cannot_happen(self):
        """Der Fall, den die Gleichverteilung ausschliesst.

        Bis zum Anschlag gefuellt haette diese Nutzlast eine Restrunde von
        16 Byte ergeben -- ein Paket, bei drei Raengen ein Abbruch im Wirt.
        """
        plan = self._pruefe(25153536 + 16)
        self.assertEqual(len(plan), 2)
        self.assertTrue(all(laenge // 16 >= WELT for _, laenge in plan))

    def test_the_round_count_is_group_uniform(self):
        """Die Graph-Bedingung. Sie haengt an nbytes, chunk_max und welt --
        an nichts, was je Rang verschieden ist."""
        nbytes = _bytes(8192)
        zahlen = {len(ar_plan(nbytes, CHUNK_MAX, WELT)) for _ in range(8)}
        self.assertEqual(zahlen, {4})

    def test_smallest_and_largest_slot(self):
        self.assertEqual(len(ar_plan(48, 16, 3)), 1)
        self.assertEqual(len(ar_plan(1 << 20, 1 << 30, 2)), 1)

    def test_rejects_nonsense(self):
        for schlecht in ((16, 0, 3), (16, CHUNK_MAX, 1), (-16, CHUNK_MAX, 3),
                         (17, CHUNK_MAX, 3)):
            with self.assertRaises(ValueError, msg=str(schlecht)):
                ar_plan(*schlecht)
        self.assertEqual(ar_plan(0, CHUNK_MAX, 3), [])


class TestArPlanAgainstAReference(CustomTestCase):
    """Replay the round loop and compare byte for byte.

    Models what ``htccl_all_reduce`` does: every round reduces its slice, the
    slices are written back into the result at their own offset. The
    reference is the element-wise sum over all ranks of the whole buffer.
    """

    def _simulate(self, nbytes, welt, chunk_max):
        daten = {
            r: [((r * 31 + i * 7) % 251) for i in range(nbytes // 4)]
            for r in range(welt)
        }
        # 0xEE-Analogon: was keine Runde beruehrt, bleibt erkennbar falsch.
        ergebnis = [-1] * (nbytes // 4)
        for versatz, laenge in ar_plan(nbytes, chunk_max, welt):
            a, b = versatz // 4, (versatz + laenge) // 4
            for i in range(a, b):
                ergebnis[i] = sum(daten[r][i] for r in range(welt))
        soll = [sum(daten[r][i] for r in range(welt)) for i in range(nbytes // 4)]
        return ergebnis, soll

    def test_one_round(self):
        ist, soll = self._simulate(_bytes(2048), WELT, CHUNK_MAX)
        self.assertEqual(ist, soll)

    def test_many_rounds(self):
        for token in (2457, 4096, 8192):
            ist, soll = self._simulate(_bytes(token), WELT, CHUNK_MAX)
            self.assertEqual(ist, soll, msg=f"{token} Token")

    def test_small_geometry_many_rounds(self):
        """Kleine Zahlen, damit die Rundenlogik selbst geprueft wird und
        nicht nur die Arithmetik grosser Vielfacher."""
        ist, soll = self._simulate(4096, 3, 64)
        self.assertEqual(ist, soll)

    def test_no_element_is_written_twice_or_skipped(self):
        for nbytes in (4096, 4096 + 16, 65536):
            plan = ar_plan(nbytes, 64, 3)
            beruehrt = []
            for versatz, laenge in plan:
                beruehrt.extend(range(versatz, versatz + laenge))
            self.assertEqual(beruehrt, list(range(nbytes)), msg=str(nbytes))


class TestA2aRounds(CustomTestCase):
    """The same answer for all_to_all, from the group-wide largest block."""

    def test_a_block_inside_the_slot_is_one_round(self):
        self.assertEqual(a2a_runden(SCHLITZ, SCHLITZ), 1)
        self.assertEqual(a2a_runden(1, SCHLITZ), 1)

    def test_a_block_over_the_slot_is_split(self):
        self.assertEqual(a2a_runden(SCHLITZ + 1, SCHLITZ), 2)
        self.assertEqual(a2a_runden(SCHLITZ * 4, SCHLITZ), 4)

    def test_zero_is_still_one_round(self):
        """Alle Bloecke leer heisst trotzdem: die Sperre wird gefahren."""
        self.assertEqual(a2a_runden(0, SCHLITZ), 1)

    def test_the_count_comes_from_the_group_wide_maximum(self):
        """Aus der eigenen Zeile gerechnet waere sie rangabhaengig -- und
        ein Rang mit einer Runde weniger ist ein Haenger, kein Fehler."""
        eigene_zeilen = [SCHLITZ // 2, SCHLITZ * 3, SCHLITZ]
        gruppenweit = max(eigene_zeilen)
        zahlen = {a2a_runden(gruppenweit, SCHLITZ) for _ in eigene_zeilen}
        self.assertEqual(zahlen, {3})

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            a2a_runden(16, 0)
        with self.assertRaises(ValueError):
            a2a_runden(-1, 16)

    def test_the_gate_follows_the_round_cap(self):
        t = _stub(a2a_max_runden=4, a2a_schlitz=1024)
        self.assertTrue(t.traegt_a2a(1024 * 4))
        self.assertFalse(t.traegt_a2a(1024 * 4 + 1))

    def test_the_transport_reports_the_count_for_the_seam(self):
        t = _stub(a2a_schlitz=1024)
        self.assertEqual(t.a2a_runden_fuer(1024 * 3), 3)


class TestHandlesGate(CustomTestCase):
    """Coverage before and after, at the sizes the analysis names."""

    def test_the_working_point_is_covered_as_before(self):
        t = _stub()
        self.assertTrue(t.handles("all_reduce", _bytes(2048)))
        self.assertEqual(t.ar_runden(_bytes(2048)), 1)

    def test_over_the_tipping_point_is_now_covered(self):
        """VORHER: handles() -> False und ein stiller Rueckfall."""
        t = _stub()
        for token in (KIPP_TOKEN + 1, 4096, 8192):
            self.assertTrue(
                t.handles("all_reduce", _bytes(token)), msg=f"{token} Token"
            )

    def test_the_coverage_is_gapless_up_to_the_round_cap(self):
        t = _stub()
        je_runde = (CHUNK_MAX // 16) * WELT * 16
        self.assertTrue(t.handles("all_reduce", je_runde * 16))
        self.assertFalse(t.handles("all_reduce", je_runde * 16 + 16))

    def test_the_hard_host_conditions_still_reject(self):
        """Was der Wirt nicht faehrt, faehrt auch die Naht nicht."""
        t = _stub()
        self.assertFalse(t.handles("all_reduce", 4096 + 1))    # kein Vielfaches von 16
        self.assertFalse(t.handles("all_reduce", 32))          # unter min_bytes
        winzig = _stub(min_bytes=16)
        self.assertFalse(winzig.handles("all_reduce", 32))     # < ein Paket je Rang

    def test_a_round_that_would_not_fit_the_window_is_refused(self):
        t = _stub(_fenster_minimum=1 << 20)
        self.assertFalse(t.handles("all_reduce", _bytes(2048)))

    def test_the_other_ops_are_unaffected(self):
        t = _stub()
        self.assertFalse(t.handles("reduce_scatter", 65536))
        self.assertTrue(t.handles("all_gather", 65536))
        self.assertTrue(t.handles("broadcast", 128))


class TestLoudFallbackNotice(CustomTestCase):
    """Rueckfall ist erlaubt, lautlos nicht.

    Outside a capture the gloo plane is a slower but usable path, so nothing
    aborts. But a transport that quietly steps aside for one size while the
    log says "transport=bar1" invalidates every number taken afterwards --
    which is exactly what would have happened in prefill.
    """

    def _comm(self, transport, gruppe="tp:0"):
        from sglang.srt.distributed.device_communicators.htccl import (
            HTCCLCommunicator,
        )

        c = HTCCLCommunicator.__new__(HTCCLCommunicator)
        c.transport = transport
        c._path_dispatcher = None
        c._rueckfall_gemeldet = set()
        c.gruppe = gruppe
        return c

    def _select(self, c, op, nbytes):
        from sglang.srt.distributed.device_communicators import htccl as mod

        with mock.patch.object(mod, "graph_erfassung_laeuft", lambda: False):
            return mod.HTCCLCommunicator._select(c, op, nbytes)

    def test_an_uncovered_size_is_announced_with_op_bytes_and_reason(self):
        c = self._comm(_stub())
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.htccl", level="WARNING"
        ) as protokoll:
            self.assertIsNone(self._select(c, "reduce_scatter", 65536))
        text = "\n".join(protokoll.output)
        self.assertIn("reduce_scatter", text)
        self.assertIn("65536", text)
        self.assertIn("tp:0", text)
        self.assertIn("Rueckfall", text)
        self.assertIn("Grund:", text)

    def test_a_covered_size_says_nothing(self):
        """Die Negativkontrolle. Ein Hinweis, der immer kommt, ist keiner."""
        c = self._comm(_stub())
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.htccl"
        )
        with mock.patch.object(logger, "warning") as warnung:
            self.assertIsNotNone(self._select(c, "all_reduce", _bytes(2048)))
            self.assertIsNotNone(self._select(c, "all_reduce", _bytes(8192)))
            self.assertIsNotNone(self._select(c, "broadcast", 128))
        warnung.assert_not_called()

    def test_the_size_that_used_to_fall_through_silently_is_now_covered(self):
        """Der Anlassfall: 4096 Token Prefill. Kein Hinweis, weil kein
        Rueckfall -- das ist der Punkt der Runden."""
        c = self._comm(_stub())
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.htccl"
        )
        with mock.patch.object(logger, "warning") as warnung:
            self.assertIsNotNone(self._select(c, "all_reduce", _bytes(4096)))
        warnung.assert_not_called()

    def test_it_speaks_once_per_op_and_size_class(self):
        """Im heissen Pfad laufen dieselben Groessen tausendfach."""
        c = self._comm(_stub())
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.htccl"
        )
        with mock.patch.object(logger, "warning") as warnung:
            for _ in range(50):
                self._select(c, "reduce_scatter", 65536)
            self.assertEqual(warnung.call_count, 1)
            # Eine andere Groessenklasse ist ein neuer Betriebspunkt.
            self._select(c, "reduce_scatter", 65536 * 64)
            self.assertEqual(warnung.call_count, 2)

    def test_each_group_speaks_for_itself(self):
        """tp und dcp bekommen verschieden grosse Fenster -- was in der
        einen passt, muss in der anderen nicht."""
        logger = logging.getLogger(
            "sglang.srt.distributed.device_communicators.htccl"
        )
        with mock.patch.object(logger, "warning") as warnung:
            for gruppe in ("tp:0", "dcp:0"):
                self._select(self._comm(_stub(), gruppe), "reduce_scatter", 65536)
            self.assertEqual(warnung.call_count, 2)

    def test_under_capture_the_bar_still_wins(self):
        """Unter Aufzeichnung gibt es keinen Rueckfall, also auch keinen
        Hinweis darauf -- dort bricht es ab, und das bleibt so."""
        from sglang.srt.distributed.device_communicators import htccl as mod

        c = self._comm(_stub())
        with mock.patch.object(mod, "graph_erfassung_laeuft", lambda: True):
            with self.assertRaises(RuntimeError):
                mod.HTCCLCommunicator._select(c, "reduce_scatter", 65536)


class TestWarumNicht(CustomTestCase):
    """The reason text. Diagnostic only -- it decides nothing."""

    def test_a_covered_size_has_no_reason(self):
        t = _stub()
        self.assertEqual(t.warum_nicht("all_reduce", _bytes(2048)), "")

    def test_an_unaligned_payload_says_so(self):
        t = _stub()
        self.assertIn("Vielfaches von 16", t.warum_nicht("all_reduce", 4097))

    def test_too_many_rounds_names_the_cap(self):
        t = _stub(ar_max_runden=2)
        grund = t.warum_nicht("all_reduce", _bytes(8192))
        self.assertIn("Runden", grund)
        self.assertIn("erlaubt sind 2", grund)

    def test_an_op_outside_the_coverage_says_so(self):
        self.assertIn("HTCCL_OPS", _stub().warum_nicht("reduce_scatter", 4096))

    def test_a_transport_that_is_not_up_says_so(self):
        t = _stub(_auf=False)
        self.assertIn("nicht aufgebaut", t.warum_nicht("all_reduce", 4096))

    def test_a_switched_off_op_says_which_switch(self):
        t = _stub(bc_an=False)
        self.assertIn("SGLANG_HTCCL_BAR1_BC", t.warum_nicht("broadcast", 128))


if __name__ == "__main__":
    unittest.main()
