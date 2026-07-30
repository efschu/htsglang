"""Die drei Fixes aus der Hebel-Messung zu #293, Schritt 2.

Der Messlauf (``docs/dev/INTEGRATION_R3_VALIDATION.md``, Abschnitt "#293
Schritt 2: Hebel-Messung gegen die Prefill-Decke") hat drei Posten benannt
und beziffert. Dieser Aufbau haelt die Antworten darauf fest, und zwar
rechnend statt argumentierend -- alles hier ist reine Arithmetik und
Zustandsfuehrung, ohne Karte.

1. **Der gitter-Vorbehalt** legte den cooperative Start waehrend jeder
   CUDA-Graph-Aufzeichnung auf die langsamere ``1blk``-Variante. Unter dem
   Prefill-Graphen kostete das 16,1 % Durchsatz bei acht Sessions
   (1334,5 -> 1151,6 tok/s; der Falsifikator mit
   ``SGLANG_HTCCL_BAR1_GRAPH_GITTER=1`` holte 1337,2). Die Vorgabe kommt
   jetzt aus ``SGLANG_HTCCL_GRAPH_FREIGABE`` -- dieselbe Frage, dasselbe
   Tor.

2. **Der Pipe-Bereich** hat dem all_reduce-Schlitz ein Viertel weggenommen
   (8188 -> 6140 KiB, Kipp-Punkt 2456 -> 1842 Token, also unter den
   Arbeitspunkt 2048). Der Messbericht schrieb das dem Ergebnisring zu; das
   war falsch, und die Rechnung hier zeigt beides: der Ring war in jenem Arm
   gar nicht da (``PIPE_DIREKT=0``), und die 6140 fallen exakt aus dem
   zusaetzlichen Schlitzsatz der Pipe.

3. **Der Ergebnisring** brach den graphfesten Direktmodus im
   Aufzeichnungs-WARMUP ab, und ``SGLANG_HTCCL_BAR1_PIPE_ERG_RING`` half
   nicht, weil die eager-Zahl eine Konstante war.
"""

import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.htccl_bar1 import (
    HTCCLBar1Transport,
    geometrie,
    graph_gitter_vorgabe,
    max_nutzlast,
)
from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
    ERG_EAGER_PLAETZE,
    erg_aufteilung,
    erg_eager_freier_platz,
    erg_eager_platz,
    erg_eager_slack,
    erg_graph_platz,
    pipe_bereich_bytes,
    pipe_fensterbedarf,
    pipe_plan,
    pipe_schlitz_vorgabe,
)
from sglang.srt.distributed.parallel_state import graph_freigabe_gesetzt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: Die Geometrie des Messlaufs: 96 MiB Fenster je Rang, drei Raenge, a2a an,
#: Pipe-Tiefe T=4, Chunkziel 1 MiB. Jede Zahl in diesem Aufbau faellt aus
#: diesen fuenf.
FENSTER = 96 << 20
WELT = 3
TIEFE = 4
CHUNK_ZIEL = 1 << 20
K_MAX = 64
#: Verborgene Breite x Elementgroesse des Modells -- damit ein Kipp-Punkt in
#: TOKEN gelesen werden kann, so wie der Messbericht ihn aufschreibt.
BYTE_JE_TOKEN = 5120 * 2
#: Der Arbeitspunkt des Laufs.
ARBEITSPUNKT_TOKEN = 2048


def _kipp_token(max_bytes: int) -> int:
    """Groesster Batch, den EINE Runde noch traegt."""
    return max_bytes // BYTE_JE_TOKEN


# ===========================================================================
# Fix 1: der gitter-Vorbehalt haengt an der Graph-Freigabe
# ===========================================================================


class TestGitterVorgabeFolgtDerFreigabe(CustomTestCase):
    """Ein Tor, eine Antwort -- nicht zwei Schalter fuer dieselbe Frage.

    ``bar1_graph_check.py`` beantwortet mit seinem Fall ``gitter`` genau die
    Frage, ob ``cudaLaunchCooperativeKernel`` sich hier aufzeichnen laesst.
    Solange daneben ein eigener Opt-in stand, konnte das Tor bestehen, die
    Freigabe stehen -- und der Kern trotzdem auf ``1blk`` zurueckfallen.
    Genau das ist passiert und hat 16,1 % gekostet.
    """

    def test_without_the_release_the_reservation_stands(self):
        self.assertFalse(graph_gitter_vorgabe({}))
        self.assertFalse(
            graph_gitter_vorgabe({"SGLANG_HTCCL_GRAPH_FREIGABE": "0"})
        )

    def test_the_release_carries_the_default(self):
        self.assertTrue(
            graph_gitter_vorgabe({"SGLANG_HTCCL_GRAPH_FREIGABE": "1"})
        )

    def test_the_override_wins_in_both_directions(self):
        """Beide Richtungen, weil beide gebraucht werden.

        Der Gate-Fall ``gitter`` faehrt den cooperative Start OHNE Freigabe
        (er soll sie ja erst begruenden); der Gate-Fall ``vorbehalt`` faehrt
        den Rueckfall MIT stehender Freigabe. Ein Schalter, der nur in eine
        Richtung wirkt, macht einen der beiden Faelle unpruefbar.
        """
        self.assertTrue(
            graph_gitter_vorgabe({"SGLANG_HTCCL_BAR1_GRAPH_GITTER": "1"})
        )
        self.assertFalse(
            graph_gitter_vorgabe({
                "SGLANG_HTCCL_GRAPH_FREIGABE": "1",
                "SGLANG_HTCCL_BAR1_GRAPH_GITTER": "0",
            })
        )
        self.assertTrue(
            graph_gitter_vorgabe({
                "SGLANG_HTCCL_GRAPH_FREIGABE": "0",
                "SGLANG_HTCCL_BAR1_GRAPH_GITTER": "1",
            })
        )

    def test_the_two_readers_of_the_release_agree_word_for_word(self):
        """Dieselbe Variable, gelesen an zwei Stellen -- also derselbe Satz.

        ``parallel_state.graph_freigabe_gesetzt`` entscheidet, ob bar1
        ueberhaupt aufgezeichnet werden darf; die Vorgabe hier entscheidet,
        WIE. Lesen die beiden ein ``"aus"`` oder ein leeres Feld
        verschieden, faellt der Unterschied erst als Durchsatzverlust auf.
        """
        import os

        for wert in ("", "0", "1", "nein", "aus", "false", "ja", "wahr", "2"):
            with mock.patch.dict(
                os.environ, {"SGLANG_HTCCL_GRAPH_FREIGABE": wert}, clear=False
            ):
                self.assertEqual(
                    graph_gitter_vorgabe(),
                    graph_freigabe_gesetzt(),
                    f"Wert {wert!r} wird verschieden gelesen",
                )

    def test_an_unset_variable_is_not_the_same_as_a_zero(self):
        """Der Unterschied, an dem die Uebersteuerung haengt.

        ``os.environ.get(name)`` gibt ``None`` fuer "gar nicht gesetzt" und
        ``""`` fuer "leer gesetzt". Nur das erste darf auf die Freigabe
        durchgreifen -- sonst waere jede leere Zuweisung ein stilles
        Einschalten.
        """
        self.assertTrue(
            graph_gitter_vorgabe({"SGLANG_HTCCL_GRAPH_FREIGABE": "1"})
        )
        self.assertFalse(
            graph_gitter_vorgabe({
                "SGLANG_HTCCL_GRAPH_FREIGABE": "1",
                "SGLANG_HTCCL_BAR1_GRAPH_GITTER": "",
            })
        )


class TestKernWaehltNachDerVorgabe(CustomTestCase):
    """Und die Vorgabe kommt auch wirklich bis in ``_kern``."""

    def _stub(self, graph_gitter: bool):
        t = HTCCLBar1Transport.__new__(HTCCLBar1Transport)
        t.graph_gitter = graph_gitter
        t._graph_gitter_gemeldet = False
        return t

    def _mit_erfassung(self):
        return mock.patch(
            "sglang.srt.distributed.device_communicators.htccl."
            "graph_erfassung_laeuft",
            lambda: True,
        )

    def test_below_the_threshold_nothing_changes(self):
        for gitter in (False, True):
            t = self._stub(gitter)
            with self._mit_erfassung():
                self.assertEqual(t._kern(1024, 4 << 20, "all_reduce"), 0)

    def test_capture_with_the_release_keeps_the_cooperative_launch(self):
        t = self._stub(True)
        with self._mit_erfassung():
            self.assertEqual(t._kern(8 << 20, 4 << 20, "all_reduce"), 1)

    def test_capture_without_it_still_falls_back_and_says_so(self):
        t = self._stub(False)
        with self._mit_erfassung():
            self.assertEqual(t._kern(8 << 20, 4 << 20, "all_reduce"), 0)
        self.assertTrue(t._graph_gitter_gemeldet)


# ===========================================================================
# Fix 2: wer dem Fenster wirklich den Platz nimmt
# ===========================================================================


class TestWerDenSchlitzKlaut(CustomTestCase):
    """Die Zurechnung des Messberichts, nachgerechnet statt uebernommen.

    Der Bericht schrieb den Verlust dem Ergebnisring zu. Der Pipe-Arm fuhr
    aber ``SGLANG_HTCCL_BAR1_PIPE_DIREKT=0``, und damit ist der Ring null
    (``htccl_bar1.py``, "if not self.pipe_an or not self.pipe_direkt").
    Uebrig bleibt genau eine Ursache, und sie trifft die Zahlen des Berichts
    auf das Byte.
    """

    def test_the_measured_slot_without_the_pipe_is_reproduced(self):
        n = max_nutzlast(WELT, FENSTER, True, False, 0)
        geo = geometrie(WELT, n, True, False, 0)
        self.assertEqual(geo["chunk_max"] // 1024, 8188)
        self.assertEqual(_kipp_token(n), 2456)

    def test_the_measured_loss_comes_from_the_pipe_set_not_from_the_ring(self):
        """6140 KiB faellt bei ``erg_ring = 0`` -- der Ring war nicht dabei."""
        n = max_nutzlast(WELT, FENSTER, True, True, 0)
        geo = geometrie(WELT, n, True, True, 0)
        self.assertEqual(geo["chunk_max"] // 1024, 6140)
        self.assertEqual(_kipp_token(n), 1842)
        self.assertLess(
            _kipp_token(n), ARBEITSPUNKT_TOKEN,
            "der Kipp-Punkt muss UNTER dem Arbeitspunkt liegen -- sonst "
            "erklaert er die zwei Runden des 20-MiB-all_reduce nicht",
        )

    def test_the_ring_is_not_reserved_without_the_direct_mode(self):
        """Die Behauptung "der Ring nimmt sich immer seinen Anteil", geprueft.

        Ohne Direkt-Modus gibt es keinen Ergebnisring in der Geometrie: kein
        Versatz, keine Schrittweite, keine Laenge. Der Regressionswaechter
        dazu, weil ein Ring, der ohne Nutzung reserviert wuerde, genau der
        Fehler waere, den der Messbericht vermutet hat.
        """
        geo = geometrie(WELT, 8 << 20, True, True, 0)
        self.assertEqual(geo["erg_ring"], 0)
        self.assertEqual(geo["off_erg"], -1)
        self.assertEqual(geo["erg_stride"], 0)
        # Und mit Direkt-Modus kostet er, was er kostet -- das ist kein
        # Fehler, sondern der Preis des Modus.
        mit = geometrie(WELT, 8 << 20, True, True, 2)
        self.assertGreater(mit["region_bytes"], geo["region_bytes"])

    def test_the_pipe_area_is_exactly_the_difference(self):
        """Die Differenz der beiden Nenner, ohne Umweg ueber die Fixpunkte.

        12 Schlitze gegen 16: netz 4, ring 4, a2a 4, Pipe 4 (je 2(R-1)).
        Beide Zahlen des Messberichts fallen aus derselben Division.
        """
        seite = 4096
        for nenner, erwartet in ((12, 8188), (16, 6140)):
            self.assertEqual(
                ((FENSTER - seite) // nenner // seite) * seite // 1024,
                erwartet,
            )
        self.assertEqual(
            geometrie(WELT, max_nutzlast(WELT, FENSTER, True, False, 0),
                      True, False, 0)["chunk_max"],
            ((FENSTER - seite) // 12 // seite) * seite,
        )
        self.assertEqual(
            geometrie(WELT, max_nutzlast(WELT, FENSTER, True, True, 0),
                      True, True, 0)["chunk_max"],
            ((FENSTER - seite) // 16 // seite) * seite,
        )


class TestPipeBereichNachBedarf(CustomTestCase):
    """Der Bereich, den die Pipe wirklich braucht -- und was er zurueckholt."""

    def _bereich(self) -> int:
        return pipe_bereich_bytes(
            WELT, TIEFE, pipe_schlitz_vorgabe(WELT, CHUNK_ZIEL)
        )

    def test_the_need_is_a_property_of_the_chunk_target_not_of_the_window(self):
        """Die Entkopplung, an der die ganze Rechnung haengt.

        Der Bedarf haengt an ``pipe_chunk_bytes``, ``T`` und ``R``. An
        nichts, was aus dem Fenster folgt -- deshalb ist er in der
        Fixpunktrechnung von ``max_nutzlast`` eine Konstante und kein
        weiterer Nenner.
        """
        for fenster in (64 << 20, 96 << 20, 256 << 20, 8 << 30):
            n = max_nutzlast(WELT, fenster, True, True, 0, self._bereich())
            geo = geometrie(WELT, n, True, True, 0, self._bereich())
            self.assertEqual(geo["pipe_bereich"], self._bereich())

    def test_the_right_sized_area_lifts_the_tipping_point_over_the_working_point(self):
        n = max_nutzlast(WELT, FENSTER, True, True, 0, self._bereich())
        geo = geometrie(WELT, n, True, True, 0, self._bereich())
        self.assertEqual(geo["chunk_max"] // 1024, 7736)
        self.assertEqual(_kipp_token(n), 2320)
        self.assertGreater(_kipp_token(n), ARBEITSPUNKT_TOKEN)

    def test_pure_pipe_without_direct_keeps_almost_the_whole_slot(self):
        """"Reines Pipe behaelt den vollen Slot", so weit es geht.

        Ganz voll wird er nicht: die Pipe braucht ihre 5,3 MiB wirklich, und
        die fehlen dem Schlitz. Von den 2048 KiB, die der volle Schlitzsatz
        gekostet hat, kommen 1596 zurueck -- 78 %.
        """
        ohne = geometrie(
            WELT, max_nutzlast(WELT, FENSTER, True, False, 0), True, False, 0
        )["chunk_max"]
        alt = geometrie(
            WELT, max_nutzlast(WELT, FENSTER, True, True, 0), True, True, 0
        )["chunk_max"]
        neu = geometrie(
            WELT, max_nutzlast(WELT, FENSTER, True, True, 0, self._bereich()),
            True, True, 0, self._bereich(),
        )["chunk_max"]
        self.assertLess(alt, neu)
        self.assertLess(neu, ohne)
        self.assertGreater((neu - alt) / (ohne - alt), 0.75)

    def test_the_eager_direct_slots_are_charged_to_the_direct_mode(self):
        """Zwei Plaetze bleiben zwei Plaetze -- sie gehoeren dem Direktmodus.

        Der eager-Pfad des Direktmodus braucht seine Plaetze wirklich; sie
        dem Pipe-Bereich abzuziehen hiesse, den Direktmodus zum Nulltarif
        auszuweisen. Was er kostet, kostet er.
        """
        ber = self._bereich()
        ohne_ring = max_nutzlast(WELT, FENSTER, True, True, 0, ber)
        mit_ring = max_nutzlast(WELT, FENSTER, True, True, 2, ber)
        self.assertLess(mit_ring, ohne_ring)

    def test_every_payload_the_window_carries_finds_a_chunk_count(self):
        """Ein knapper Schlitz darf Deckung kosten -- hier kostet er keine.

        ``pipe_plan`` sucht sein K aufsteigend; passt kein K, meldet sich der
        Weg ueber ``handles()`` ab. Das waere richtig, aber teuer. Geprueft
        wird deshalb, dass es zwischen der Untergrenze der Pipe und der
        groessten Nutzlast des Fensters gar nicht erst dazu kommt.
        """
        schlitz = pipe_schlitz_vorgabe(WELT, CHUNK_ZIEL)
        maxb = max_nutzlast(WELT, FENSTER, True, True, 0, self._bereich())
        schritt = 16 * 997
        nb = 256 << 10
        geprueft = 0
        while nb <= maxb:
            self.assertIsNotNone(
                pipe_plan(nb, WELT, schlitz, TIEFE, 0, CHUNK_ZIEL, K_MAX),
                f"{nb} Byte traegt der Schlitz von {schlitz} Byte nicht",
            )
            geprueft += 1
            nb += schritt
        self.assertGreater(geprueft, 1000)

    def test_what_the_kernel_touches_stays_inside_what_the_layout_reserves(self):
        """Die eine Bedingung, die ein zu kleiner Bereich verletzen wuerde.

        Der Kern faehrt zwei Ringe zu je ``T(R-1)`` Schlitzen; die hoechste
        Adresse ist ``2 T (R-1) * schlitz`` ab ``off_pipe``. Der reservierte
        Bereich ist dieselbe Zahl, auf eine Seite aufgerundet -- und der
        Ergebnisring beginnt erst dahinter.
        """
        for welt in (2, 3, 4, 8):
            for tiefe in (2, 4, 8):
                for ziel in (256 << 10, 1 << 20, 4 << 20):
                    schlitz = pipe_schlitz_vorgabe(welt, ziel)
                    ber = pipe_bereich_bytes(welt, tiefe, schlitz)
                    self.assertGreaterEqual(
                        ber, pipe_fensterbedarf(welt, tiefe, schlitz)
                    )
                    self.assertEqual(ber % 4096, 0)
                    self.assertEqual(schlitz % 16, 0)
                    n = max_nutzlast(welt, FENSTER, True, True, 2, ber)
                    if n <= 0:
                        continue
                    geo = geometrie(welt, n, True, True, 2, ber)
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
        """Der Falsifikator: die Pruefung oben muss Zaehne haben.

        Der plausible Fehler ist ein Ergebnisring, der direkt hinter
        ``off_pipe`` angehaengt wird, weil jemand den Bereich vergisst. Er
        landet dann mitten in den Pipe-Schlitzen.
        """
        ber = self._bereich()
        n = max_nutzlast(WELT, FENSTER, True, True, 2, ber)
        geo = geometrie(WELT, n, True, True, 2, ber)
        falsch = geo["off_pipe"]
        self.assertLess(
            falsch,
            geo["off_pipe"] + pipe_fensterbedarf(
                WELT, TIEFE, pipe_schlitz_vorgabe(WELT, CHUNK_ZIEL)
            ),
        )
        self.assertNotEqual(falsch, geo["off_erg"])

    def test_the_old_cut_is_still_available_and_byte_identical(self):
        """``pipe_bereich = 0`` heisst "wie frueher" -- und zwar genau so."""
        for max_bytes in (64 << 10, 8 << 20):
            alt = geometrie(WELT, max_bytes, True, True, 2, 0)
            self.assertEqual(
                alt["pipe_bereich"],
                2 * (WELT - 1) * alt["chunk_max"],
            )
            self.assertEqual(
                alt["off_erg"], alt["off_pipe"] + alt["pipe_bereich"]
            )


# ===========================================================================
# Fix 3: der eager-Teil des Ergebnisrings
# ===========================================================================


class TestErgAufteilungIstEinstellbar(CustomTestCase):
    def test_the_default_is_unchanged(self):
        self.assertEqual(ERG_EAGER_PLAETZE, 2)
        self.assertEqual(erg_aufteilung(5, True), (2, 3))
        self.assertEqual(erg_aufteilung(5, False), (5, 0))

    def test_a_bigger_ring_used_to_hand_out_graph_slots_only(self):
        """Der Befund des Messlaufs, als Zahl.

        ``ERG_RING=5`` gab drei GRAPH-Plaetze und liess die eager-Zahl bei
        zwei -- und der Abbruch fiel im Aufzeichnungs-WARMUP an, das eager
        laeuft. Deshalb konnte der Knopf nicht wirken, und ``ERG_RING=50``
        haette es auch nicht.
        """
        for ring in (5, 8, 50):
            eager, _graph = erg_aufteilung(ring, True)
            self.assertEqual(eager, 2)

    def test_now_the_eager_share_is_the_parameter(self):
        self.assertEqual(erg_aufteilung(5, True, 4), (4, 1))
        self.assertEqual(erg_aufteilung(5, True, 5), (5, 0))
        self.assertEqual(erg_aufteilung(5, True, 9), (5, 0))
        self.assertEqual(erg_aufteilung(5, False, 4), (5, 0))

    def test_the_graph_supply_still_begins_behind_the_eager_share(self):
        eager, graph = erg_aufteilung(7, True, 4)
        self.assertEqual((eager, graph), (4, 3))
        self.assertEqual(
            [erg_graph_platz(v, eager, graph) for v in range(4)],
            [4, 5, 6, None],
        )


class TestFreierEagerPlatz(CustomTestCase):
    def test_with_everything_free_the_rotation_is_the_old_one(self):
        voriger = -1
        folge = []
        for _ in range(6):
            voriger = erg_eager_freier_platz(voriger, 3, [False] * 3)
            folge.append(voriger)
        self.assertEqual(folge, [0, 1, 2, 0, 1, 2])
        self.assertEqual(
            folge[:3],
            [erg_eager_platz(-1, 3), erg_eager_platz(0, 3),
             erg_eager_platz(1, 3)],
        )

    def test_a_held_slot_is_skipped_instead_of_aborting(self):
        self.assertEqual(erg_eager_freier_platz(-1, 2, [True, False]), 1)
        self.assertEqual(erg_eager_freier_platz(0, 3, [False, True, False]), 2)

    def test_all_held_means_no_slot_and_that_is_an_answer(self):
        self.assertIsNone(erg_eager_freier_platz(-1, 2, [True, True]))
        self.assertIsNone(erg_eager_freier_platz(1, 4, [True] * 4))

    def test_a_ring_without_slots_is_a_programming_error_not_a_none(self):
        with self.assertRaises(ValueError):
            erg_eager_freier_platz(-1, 0, [])


class TestSlackIstEineUntergrenze(CustomTestCase):
    """Ein zu GROSSER Slack waere die schwaechere Wartebedingung.

    Der Kern wartet darauf, dass der Peer Generation ``ZIEL - slack + 1``
    betreten hat. Je groesser der Slack, desto frueher greift die Bedingung -- die gefaehrliche Richtung. Er muss deshalb den tatsaechlichen
    Wiederverwendungsabstand unterschaetzen duerfen, nie ueberschaetzen.
    """

    def test_strict_rotation_gives_exactly_the_number_of_slots(self):
        L = 3
        zuletzt = [None] * L
        zaehler = 0
        gesehen = []
        for _ in range(3 * L):
            i = erg_eager_freier_platz(
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
        # Platz 0 bei Aufruf 0, dann bei Aufruf 1 wieder Platz 0.
        zuletzt[0] = 0
        self.assertEqual(erg_eager_slack(0, 1, zuletzt, L), 1)
        self.assertEqual(erg_eager_slack(0, 2, zuletzt, L), 2)
        self.assertEqual(erg_eager_slack(0, 9, zuletzt, L), L)

    def test_an_unused_slot_may_take_the_full_distance(self):
        self.assertEqual(erg_eager_slack(1, 7, [None, None], 2), 2)

    def test_the_slack_is_never_zero(self):
        """``0`` schaltet den Handschlag im Kern ganz ab."""
        for zaehler in range(5):
            self.assertGreaterEqual(
                erg_eager_slack(0, zaehler, [zaehler], 4), 1
            )


def _stub(**kw):
    """Ein Transport ohne ``__init__`` -- nur die Felder von ``_erg_platz``."""
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
        "graph_erfassung_laeuft",
        lambda: False,
    )


class TestEagerVollFaelltZurueckStattAbzubrechen(CustomTestCase):
    """Der Boot-Stopper des Direktmodus, an der Naht.

    Vorher warf ``_erg_platz`` hier ``Bar1Unverfuegbar`` -- im
    Aufzeichnungs-WARMUP also ein toter Server. Der Abbruch war die falsche
    Antwort auf die richtige Sorge: verboten bleibt das Beschreiben eines
    gehaltenen Puffers, und das passiert bei ``direkt=0`` gerade nicht. Es
    bleibt derselbe Weg, den der erschoepfte Graph-Vorrat ein paar Zeilen
    weiter oben schon nimmt.
    """

    def test_a_held_slot_no_longer_kills_the_call(self):
        t = _stub()
        with _ohne_erfassung():
            behalten, _platz, _slack = t._erg_platz(object())
            zweiter = t._erg_platz(object())
        self.assertIsNotNone(behalten)
        self.assertIsNotNone(zweiter)
        self.assertNotEqual(zweiter[1], 0)

    def test_all_slots_held_falls_back_to_direct_zero_and_counts_it(self):
        t = _stub()
        gehalten = []
        with _ohne_erfassung():
            for _ in range(2):
                gehalten.append(t._erg_platz(object()))
            self.assertIsNone(t._erg_platz(object()))
            self.assertIsNone(t._erg_platz(object()))
        self.assertEqual(t._erg_eager_voll, 2)
        self.assertTrue(t._erg_eager_voll_gemeldet)
        # Und sobald der Aufrufer loslaesst, laeuft der Direktmodus wieder.
        gehalten.clear()
        with _ohne_erfassung():
            self.assertIsNotNone(t._erg_platz(object()))

    def test_more_eager_slots_carry_more_live_results(self):
        """Der Knopf, der im Messlauf gefehlt hat, an der Naht geprueft."""
        t = _stub(
            _erg_lebt=[None] * 4,
            _erg_zuletzt=[None] * 4,
            _erg_eager_plaetze=4,
            _geo={"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 4},
        )
        gehalten = []
        with _ohne_erfassung():
            for _ in range(4):
                gehalten.append(t._erg_platz(object()))
            self.assertIsNone(t._erg_platz(object()))
        self.assertEqual([g[1] for g in gehalten], [0, 1, 2, 3])
        self.assertEqual(t._erg_eager_voll, 1)

    def test_the_handshake_slack_follows_the_real_reuse_distance(self):
        t = _stub(pipe_direkt_graph=True)
        with _ohne_erfassung():
            erster = t._erg_platz(object())      # Platz 0, wird GEHALTEN
            t._erg_platz(object())               # Platz 1, sofort verworfen
            dritter = t._erg_platz(object())     # wieder Platz 1
        self.assertEqual(erster[1], 0)
        self.assertEqual(dritter[1], 1)
        # Platz 1 lag einen Aufruf zurueck, nicht zwei.
        self.assertEqual(dritter[2], 1)

    def test_the_measured_rotation_keeps_its_old_slack(self):
        t = _stub(pipe_direkt_graph=True)
        with _ohne_erfassung():
            for _ in range(5):
                _out, _platz, slack = t._erg_platz(object())
                self.assertEqual(slack, t._erg_eager_plaetze)


if __name__ == "__main__":
    unittest.main()
