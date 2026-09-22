"""#160: KARTE UND RANG MUESSEN DIESELBE MENGE MEINEN -- eine Formel, nicht zwei.

fnFL2w133 starb an `#107: 113 eigene kalte Experten haben in der KARTE
keinen Platz`. Der sichtbare Anlass war der halbgebaute Spiegel (#159),
die STILLE Ursache aber ist aelter und trifft JEDEN Boot, auch ohne
Spiegel: die Karte und der Rang rechnen ihre Residenzgroesse mit
VERSCHIEDENEN FORMELN.

    Rang  (expert_offload.resident_slot_count):  max(1, ceil(f * n))
    Karte (expert_map.resident_sharded/_unsharded):   round(n * f)

Gemessen an den acht Fraktionen, die heute wirklich gefahren wurden,
divergieren SIEBEN. Beispiel P-Stufe 0 bei FR_P 0.459 ueber 512
Experten: der Rang haelt 236 resident, die Karte schreibt 235 auf. Die
Karte nennt Id 235 also kalt und gibt ihr einen Store-Platz, den der
Rang nie benutzt -- und in der gespiegelten Form wird daraus der
Abbruch, weil die Verschiebung die Off-by-one nicht mehr verdeckt.

Das ist [[zwei-seiten-einer-naht-fragen-dasselbe]]: beide Seiten muessen
DIESELBE Funktion mit DEMSELBEN Argument fragen. Die Karte ist die
juengere Seite, also folgt sie dem Bestandscode -- `resident_slot_count`
haengt am VRAM-Sizing (#439-Latch) und hat Konsumenten, die Karte nicht.
"""

import math
import unittest

from sglang.srt.layers.moe.expert_map import (
    bounds,
    build,
    resident_sharded,
    resident_unsharded,
    scaled_spans,
)
from sglang.srt.layers.moe.expert_offload import resident_slot_count


# Die Fraktionen, die am 22.09. wirklich gefahren wurden (Arm + Boot-Logs).
ECHTE_FAELLE = [
    ("P Stufe 0, w133", 512, 0.459),
    ("P uniform, w132", 512, 0.309),
    ("D Rang 0, w133", 192, 0.400),
    ("D Rang 1, w133", 144, 0.545),
    ("D Rang 2, w133", 176, 0.449),
    ("D Rang 1, Bestform", 137, 0.564),
    ("D Rang 2, Bestform", 168, 0.467),
]


class TestEineFormelFuerBeideSeiten(unittest.TestCase):
    def test_unsharded_zaehlt_wie_der_rang(self):
        """P: die Karte schreibt genau so viele resident wie der Rang haelt."""
        for name, total, f in [(n, t, x) for n, t, x in ECHTE_FAELLE if t == 512]:
            with self.subTest(name):
                karte = resident_unsharded([f], total)[0]
                self.assertEqual(
                    len(karte),
                    resident_slot_count(total, f),
                    f"{name}: Karte {len(karte)} != Rang "
                    f"{resident_slot_count(total, f)} -- die eine Id "
                    f"dazwischen ist im Rang resident und in der Karte kalt",
                )

    def test_sharded_zaehlt_je_rang_wie_der_rang(self):
        """D: dasselbe je Band, gegen die SKALIERTEN Spans."""
        ratios, fr = [183, 137, 168], [0.400, 0.545, 0.449]
        spans = scaled_spans(ratios, 512)
        res = resident_sharded(ratios, fr, 512)
        for i, (span, f) in enumerate(zip(spans, fr)):
            with self.subTest(rang=i, span=span, f=f):
                self.assertEqual(len(res[i]), resident_slot_count(span, f))

    def test_die_formel_wird_geteilt_nicht_nachgebaut(self):
        """Der Riegel gegen den Rueckfall: expert_map RUFT die Funktion.

        Ein Test auf Zahlengleichheit allein wuerde eine nachgebaute
        `ceil`-Kopie durchlassen -- und eine Kopie divergiert beim
        naechsten Mal wieder. Deshalb liest dieser Test die Quelle.
        """
        import inspect

        from sglang.srt.layers.moe import expert_map as em

        quelle = inspect.getsource(em)
        self.assertIn(
            "resident_slot_count",
            quelle,
            "expert_map muss die Zaehlung des Rangs AUFRUFEN, nicht nachbauen",
        )
        for fn in (em.resident_sharded, em.resident_unsharded):
            self.assertNotIn(
                "round(",
                inspect.getsource(fn),
                f"{fn.__name__} rundet noch selbst statt zu fragen",
            )

    def test_untergrenze_eins_wandert_mit(self):
        """`max(1, ...)`: FR_D[0]=0.006 ueber 183 -> der Rang haelt 2, nie 0.

        Die alte Kartenformel gab hier 1; ein Rang, der 2 haelt, hatte
        damit eine Id zuviel als 'kalt' gebucht.
        """
        self.assertEqual(len(resident_sharded([183, 137, 168], [0.006, 0.5, 0.5], 512)[0]),
                         resident_slot_count(scaled_spans([183, 137, 168], 512)[0], 0.006))

    def test_karte_bleibt_in_sich_geschlossen(self):
        """Nach dem Umbau muss die Karte weiter aufgehen: resident + slots."""
        k = build(512, [183, 137, 168], 0.459, [0.400, 0.545, 0.449])
        for phase in ("P", "D"):
            res = k["phases"][phase]["resident"]
            slot_of = k["phases"][phase]["slot_of"]
            for i, r in enumerate(res):
                überlappung = set(r) & {int(g) for g in slot_of}
                self.assertEqual(
                    überlappung, set(),
                    f"{phase}/{i}: {len(überlappung)} Ids sind resident UND im Store",
                )

    def test_mutant_round_stirbt(self):
        """Gegenprobe: mit `round` divergieren die echten Faelle wieder."""
        divergent = [
            name for name, n, f in ECHTE_FAELLE
            if round(n * f) != resident_slot_count(n, f)
        ]
        self.assertGreaterEqual(
            len(divergent), 5,
            "Wenn `round` und `ceil` hier zusammenfielen, waere dieser Test "
            "blind -- er lebt davon, dass sie es NICHT tun",
        )


if __name__ == "__main__":
    unittest.main()
