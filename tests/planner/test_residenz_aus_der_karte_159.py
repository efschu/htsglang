"""#159: DER RANG NIMMT SEINE RESIDENZ AUS DER KARTE.

`expert_map.resident_of()` existiert seit #107 und hatte NULL AUFRUFER.
Die Karte beschrieb also eine Residenz, die niemand herstellte --
`resident_fraction_for_rank()` machte weiter, was es immer tat. Ohne
Spiegel faellt das nicht auf (beide sagen "die ersten N"), mit Spiegel
stirbt der Boot sofort:

    fnFL2w133: RuntimeError: #107: 113 eigene kalte Experten haben in der
    KARTE keinen Platz (erste: [236, 237, 238, 239], Phase P, lo=0)

Das ist die vierte Instanz derselben Klasse an EINEM Tag -- #140 (Fraction
gerechnet, nie gesetzt), #156 (reserve_mib_by_stage, nie uebergeben),
#107 (FLLIPER_MOE_EXPERT_MAP gelesen, nie geschrieben) -- und diese hier
habe ich selbst gebaut. Deshalb prueft dieser Test nicht nur, DASS der
Leser richtig rechnet, sondern dass ihn auch jemand RUFT.
"""

import inspect
import unittest

from flliper.srt.layers.moe import expert_map as em
from flliper.srt.layers.moe import expert_offload as eo


class FakeLayer:
    """Das Minimum, das `_layer_expert_window` und der Leser anfassen."""

    def __init__(self, num_experts, num_local, rank=0, layer_id=0,
                 shard=False, lo=0):
        self.num_experts = num_experts
        self.num_local_experts = num_local
        self.moe_tp_rank = rank
        self.layer_id = layer_id
        self._expert_shard_generic = shard
        if shard:
            self._gguf_expert_range = (lo, lo + num_local - 1)


class TestReaderExistsAndIsCalled(unittest.TestCase):
    def test_resident_of_now_has_callers(self):
        """DIE Lehre des Tages: ein Leser ohne Aufrufer ist kein Feature.

        Gegenprobe zu [[riegel-hinter-dem-was-er-sichert]] -- dort war es
        ein Leser OHNE SCHREIBER, hier ein Leser OHNE AUFRUFER. Beide Male
        sah der Code vollstaendig aus.
        """
        source = inspect.getsource(eo)
        self.assertIn(
            "resident_of", source,
            "expert_offload ruft resident_of nicht -- die Karte beschreibt "
            "dann wieder etwas, das niemand herstellt",
        )

    def test_presplit_asks_map_before_hotset(self):
        """Echte Alternativen, kein Vorspann (#107/2)."""
        source = inspect.getsource(eo.presplit_expert_offload_after_repack)
        self.assertIn("_karten_residenz_lokal", source)
        i_card = source.index("_karten_residenz_lokal")
        i_hotset = source.index("_hotset_local_ids")
        self.assertLess(i_card, i_hotset, "die Karte muss ZUERST gefragt werden")
        self.assertIn("else:", source[i_card:i_hotset],
                      "das Hotset muss im else-Zweig stehen, nicht daneben")


class TestGlobalToLocalConversion(unittest.TestCase):
    def test_unsharded_pp_stage_local_is_global(self):
        self.assertEqual(eo._layer_expert_window(FakeLayer(512, 512)), (0, False))

    def test_expert_shard_has_window_and_pad(self):
        L = FakeLayer(512, 145, shard=True, lo=192)
        self.assertEqual(eo._layer_expert_window(L), (192, True))

    def test_ep_slice_has_no_window(self):
        """Weder unsharded noch Expertenshard -> None, alles bleibt wie bisher."""
        self.assertIsNone(eo._layer_expert_window(FakeLayer(512, 170)))

    def test_band_starts_at_scaled_boundary(self):
        """D-Rang 1 haelt globale Ids ab 192 -- NICHT ab 183 (#RATIOS).

        Die rohen Ratios 183,137,168 summieren 488; der Server skaliert auf
        192,144,176. fnFL2w49 starb an genau den fuenf Ids dazwischen.
        """
        emap = em.build(512, [183, 137, 168], 0.377, [0.006, 0.545, 0.449])
        self.assertEqual(em.resident_of(emap, "D", 1)[0], 192)

    def test_without_map_nothing_changes(self):
        """Kein Raten: ohne Karte gilt das Hotset, ohne Hotset die alte Regel."""
        self.assertIsNone(eo._karten_residenz_lokal(FakeLayer(512, 512), 512))


class TestMapIsActuallyRead(unittest.TestCase):
    def _with_map(self, tmpdir, **kw):
        import json
        import os
        from flliper.srt.layers.moe import expert_store as es

        emap = em.build(**kw)
        file_path = os.path.join(tmpdir, "karte.json")
        with open(file_path, "w") as f:
            json.dump(emap, f)
        es._EXPERT_MAP_CACHE.clear()
        os.environ["FLLIPER_MOE_EXPERT_MAP"] = file_path
        return emap

    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import os
        from flliper.srt.layers.moe import expert_store as es
        os.environ.pop("FLLIPER_MOE_EXPERT_MAP", None)
        es._EXPERT_MAP_CACHE.clear()

    def test_pp_stage_gets_exactly_the_map_set(self):
        k = self._with_map(self._tmp, total=512, ratios=[183, 137, 168],
                            fr_pp=[0.377, 0.700, 0.442],
                            fr_tp=[0.006, 0.545, 0.449])
        import os
        os.environ["FLLIPER_PDFLIP_GROUP"] = "P"
        try:
            L = FakeLayer(512, 512, rank=0)
            local_ids = eo._karten_residenz_lokal(L, 512)
            self.assertIsNotNone(local_ids, "der Leser findet die Karte nicht")
            self.assertEqual(list(local_ids), k["phases"]["P"]["resident"][0],
                             "PP: lokal == global, also 1:1 die Kartenmenge")
            # #160 zahlt sich hier aus: die Kartenmenge IST die Zahl, die
            # resident_slot_count fuer dieselbe Fraction liefert.
            self.assertEqual(len(local_ids),
                             eo.resident_slot_count(512, 0.377))
        finally:
            os.environ.pop("FLLIPER_PDFLIP_GROUP", None)

    def test_d_rank_converts_band_back_to_local(self):
        """Der schwierige Fall: Fenster ab 192 UND der Pad-Experte bei lokal 0.

        global -> lokal ist ``g - lo + 1`` bei pad. Der Pad selbst hat kein
        globales Gegenstueck (#82) und muss trotzdem resident bleiben --
        sonst faellt JEDER fremde Top-k-Treffer in den Spill-Pool.
        """
        k = self._with_map(self._tmp, total=512, ratios=[183, 137, 168],
                            fr_pp=[0.377, 0.700, 0.442],
                            fr_tp=[0.006, 0.545, 0.449])
        import os
        os.environ["FLLIPER_PDFLIP_GROUP"] = "D"
        try:
            glob = k["phases"]["D"]["resident"][1]        # Band ab 192
            L = FakeLayer(512, 145, rank=1, shard=True, lo=192)
            local_ids = eo._karten_residenz_lokal(L, 145)
            self.assertIsNotNone(local_ids)
            expected = sorted({0} | {g - 192 + 1 for g in glob})
            self.assertEqual(list(local_ids), expected)
            self.assertEqual(local_ids[0], 0, "der Pad-Experte bleibt resident")
            self.assertEqual(len(local_ids), len(glob) + 1,
                             "genau die Kartenmenge plus der Pad")
        finally:
            os.environ.pop("FLLIPER_PDFLIP_GROUP", None)

    def test_foreign_ids_are_dropped_not_crashing(self):
        """Die Karte fuehrt ALLE Raenge; fremde Baender sind kein Fehler."""
        self._with_map(self._tmp, total=512, ratios=[183, 137, 168],
                        fr_pp=[0.377, 0.700, 0.442],
                        fr_tp=[0.006, 0.545, 0.449])
        import os
        os.environ["FLLIPER_PDFLIP_GROUP"] = "D"
        try:
            # Rang 2 haelt ab 336; mit einem Fenster ab 192 liegen seine Ids
            # teils ausserhalb -- der Leser darf sie nur weglassen.
            L = FakeLayer(512, 145, rank=2, shard=True, lo=192)
            local_ids = eo._karten_residenz_lokal(L, 145)
            self.assertTrue(local_ids is None or all(0 <= e < 145 for e in local_ids))
        finally:
            os.environ.pop("FLLIPER_PDFLIP_GROUP", None)

    def test_mutant_map_does_not_match_fraction(self):
        """Die Verweigerung, die den stillen Widerspruch ersetzt."""
        k = self._with_map(self._tmp, total=512, ratios=[183, 137, 168],
                            fr_pp=[0.377, 0.700, 0.442],
                            fr_tp=[0.006, 0.545, 0.449])
        import os
        os.environ["FLLIPER_PDFLIP_GROUP"] = "P"
        try:
            L = FakeLayer(512, 512, rank=0)
            local_ids = eo._karten_residenz_lokal(L, 512)
            # Der Presplit vergleicht gegen resident_slot_count(E, frac).
            # Mit einer ANDEREN Fraction als der, mit der die Karte gebaut
            # wurde, muessen die Zahlen auseinandergehen -- genau das faengt
            # die Verweigerung ab, bevor irgendetwas alloziert wird.
            self.assertNotEqual(len(local_ids), eo.resident_slot_count(512, 0.50),
                                "sonst prueft die Verweigerung nichts")
        finally:
            os.environ.pop("FLLIPER_PDFLIP_GROUP", None)


if __name__ == "__main__":
    unittest.main()
