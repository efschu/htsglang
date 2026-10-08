"""AP-A (Profil-Planer 06.10.): the hardware profile is PERSISTED at the first start, carries the data-sheet SM count and
nominal bandwidth with their source, and the reference-rig document stays what it was.

No GPU, no NVML: NVML is injected.  What is under test: (1) the document of the three-card reference inventory equals the
document 173161c595 assembled, apart from the new fields; (2) the file is written once, kept when the cards differ, replaced
only on request, never overwritten by an empty view, and a failed write is a state, not an exception; (3) the SM count of an
unmeasured card is the data sheet's, labelled "Datasheet", a measurement still wins, and the open measurement gap stays open.
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _hwprofile_fixture_1006 as fx  # noqa: E402

from flliper.srt.rigmon import hardware_profile as hp  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: what AP-A adds to a card of the document (everything else must equal the document before AP-A)
NEW_CARD_KEYS = ("clocks", "mem_bus_bits", "catalog")


def _strip_new(doc):
    doc = copy.deepcopy(doc)
    doc.pop("id", None)
    doc.pop("created", None)
    for c in doc["cards"]:
        for k in NEW_CARD_KEYS:
            c.pop(k, None)
        c["mem_gbs"].pop("nominal", None)
    return doc


class TestReferenceProfileUnchanged(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        fx.write_probe(self.d.name)
        with open(fx.GOLDEN, encoding="utf-8") as fh:
            self.golden = json.load(fh)

    def _build(self, **kw):
        return hp.build(cache_dir=self.d.name, nvml=fx.nvml(), now=fx.NOW, **kw)

    def test_default_build_equals_the_document_before_ap_a_apart_from_the_new_fields(self):
        self.assertEqual(_strip_new(self._build()), self.golden)

    def test_without_the_data_sheet_only_the_nvml_fields_are_new(self):
        doc = self._build(datasheet=None)
        for c in doc["cards"]:
            self.assertNotIn("catalog", c)
            self.assertNotIn("nominal", c["mem_gbs"])
        self.assertEqual(_strip_new(doc), self.golden)

    def test_new_nvml_fields_say_nvml_and_unit(self):
        c = [x for x in self._build()["cards"] if x["uuid"] == fx.U1][0]
        self.assertEqual(c["clocks"]["sm_max_mhz"], {"v": 2100, "src": "NVML", "unit": "MHz"})
        self.assertEqual(c["clocks"]["mem_max_mhz"]["v"], 14000)
        self.assertEqual(c["mem_bus_bits"], {"v": 512, "src": "NVML", "unit": "bit"})

    def test_measured_sm_count_is_untouched_by_the_data_sheet(self):
        for c in self._build()["cards"]:
            self.assertEqual(c["sm_count"]["src"], "gemessen")
        self.assertEqual([c["sm_count"]["v"] for c in self._build()["cards"] if c["uuid"] == fx.U1], [170])

    def test_the_document_still_validates(self):
        self.assertEqual(hp.validate(self._build()), [])


class TestDataSheetSm(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _build(self, **kw):
        return hp.build(cache_dir=self.d.name, nvml=fx.nvml(), now=fx.NOW, **kw)

    def test_unmeasured_cards_get_the_data_sheet_sm_labelled_datenblatt(self):
        doc = self._build()          # no probe file: nothing measured
        sm = {c["nvml_index"]: c["sm_count"] for c in doc["cards"]}
        self.assertEqual(sm[1]["v"], 170)                       # hw_sim CATALOG["5090"]
        self.assertEqual(sm[0]["v"], 68)                        # 3080-10G and 3080-20G agree
        for n in sm.values():
            self.assertEqual(n["src"], "Datasheet")
            self.assertIn("hw_sim.py", n["note"])
            self.assertIn("not measured", n["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_the_sm_gap_stays_open_until_the_probe_has_read_it(self):
        doc = self._build()
        for ord_, gaps in doc["unmeasured"].items():
            self.assertIn("sm_count", gaps)
        self.assertTrue(doc["measure_needed"])

    def test_a_card_hw_sim_does_not_know_keeps_the_open_gap_as_nicht_gemessen(self):
        cards, drv, iss = fx.nvml()
        cards[0] = dict(cards[0], name="NVIDIA GeForce RTX 9999", cc=[9, 9])
        doc = hp.build(cache_dir=self.d.name, nvml=(cards, drv, iss), now=fx.NOW)
        n = [c for c in doc["cards"] if c["nvml_index"] == 0][0]["sm_count"]
        self.assertEqual((n["v"], n["src"]), (None, "nicht gemessen"))

    def test_hw_sim_datasheet_rules(self):
        self.assertEqual(hp.hw_sim_datasheet({"name": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "total_mib": 32607})["sm_count"], 170)
        # a total that matches no entry: answered only because all candidates agree (3080 10G and 20G: 68)
        self.assertEqual(hp.hw_sim_datasheet({"name": "NVIDIA GeForce RTX 3080", "cc": [8, 6], "total_mib": 12288})["sm_count"], 68)
        # same name, other cc: no match
        self.assertEqual(hp.hw_sim_datasheet({"name": "NVIDIA GeForce RTX 3080", "cc": [8, 9], "total_mib": 20480}), {})
        self.assertEqual(hp.hw_sim_datasheet({"name": "unbekannt", "cc": [8, 6], "total_mib": 1}), {})

    def test_an_injected_lookup_fills_nominal_bandwidth_and_catalog(self):
        def lookup(row):
            return {"mem_bw_gbs": 1792, "bw_note": "Catalog X", "catalog": {"id": "x", "preset": True, "origin": "Datasheet"}}

        doc = self._build(datasheet=lookup)
        c = doc["cards"][1]
        self.assertEqual(c["mem_gbs"]["nominal"], {"v": 1792, "src": "Datasheet", "unit": "GB/s", "note": "Catalog X"})
        self.assertEqual(c["catalog"]["id"], "x")
        self.assertEqual(hp.validate(doc), [])

    def test_a_card_without_a_catalog_entry_says_so(self):
        c = self._build()["cards"][0]
        self.assertEqual(c["mem_gbs"]["nominal"]["src"], "nicht gemessen")
        self.assertIn("catalog", c["mem_gbs"]["nominal"]["note"])
        self.assertIsNone(c["catalog"])

    def test_a_failing_lookup_is_an_issue_not_an_exception(self):
        def boom(row):
            raise RuntimeError("kaputt")

        doc = self._build(datasheet=boom)
        self.assertTrue(any("Datasheet lookup" in i for i in doc["sources"]["nvml"]["issues"]))
        self.assertEqual(len(doc["cards"]), 3)


class TestPersistence(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.cache = os.path.join(self.d.name, "cache")
        os.makedirs(self.cache)
        fx.write_probe(self.cache)
        self.path = os.path.join(self.d.name, "var", "lib", "flliper", "hardware.json")

    def _live(self, cards=None):
        return hp.build(cache_dir=self.cache, nvml=cards or fx.nvml(), now=fx.NOW)

    def test_the_first_call_writes_the_file_and_creates_the_directory(self):
        self.assertFalse(os.path.exists(self.path))
        r = hp.capture(self.path, live=self._live(), now=fx.NOW)
        self.assertEqual(r["state"], "erst_erfasst")
        self.assertTrue(os.path.isfile(self.path))
        doc, problem = hp.load_profile(self.path)
        self.assertIsNone(problem)
        self.assertEqual(doc["schema"], "flliper.hardware/1")
        self.assertEqual(doc["capture"], {"at": fx.NOW, "reason": "erster Start"})
        self.assertEqual([c["uuid"] for c in doc["cards"]], [c["uuid"] for c in r["live"]["cards"]])
        # nothing but the file is left in the directory (the temporary file is renamed away)
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["hardware.json"])

    def test_the_persisted_document_is_the_live_one_plus_the_stamp(self):
        live = self._live()
        hp.capture(self.path, live=live, now=fx.NOW)
        doc, _ = hp.load_profile(self.path)
        doc.pop("capture")
        self.assertEqual(doc, json.loads(json.dumps(live)))

    def test_the_second_call_finds_it_and_does_not_rewrite(self):
        hp.capture(self.path, live=self._live(), now=fx.NOW)
        before = os.stat(self.path).st_mtime_ns
        r = hp.capture(self.path, live=self._live(), now=fx.NOW + 5)
        self.assertEqual(r["state"], "vorhanden")
        self.assertTrue(r["drift"]["same"])
        self.assertEqual(os.stat(self.path).st_mtime_ns, before)

    def test_a_changed_inventory_is_reported_and_the_file_is_kept(self):
        hp.capture(self.path, live=self._live(), now=fx.NOW)
        raw = open(self.path, encoding="utf-8").read()
        r = hp.capture(self.path, live=self._live(fx.nvml(n=2, driver="600.1")), now=fx.NOW + 5)
        self.assertEqual(r["state"], "abweichend")
        ch = " | ".join(r["drift"]["changes"])
        self.assertIn(fx.U2, ch)
        self.assertIn("Driver was 595.58, now 600.1", ch)
        self.assertEqual(open(self.path, encoding="utf-8").read(), raw)
        self.assertEqual(len(r["show"]["cards"]), 2)          # the live view is what is shown

    def test_a_new_measurement_is_not_a_different_machine(self):
        hp.capture(self.path, live=self._live(), now=fx.NOW)
        fx.write_probe(self.cache, name="card_probe-new.json", created=fx.NOW - 10)
        self.assertEqual(hp.capture(self.path, live=self._live(), now=fx.NOW + 5)["state"], "vorhanden")

    def test_neu_erfassen_replaces_the_file(self):
        hp.capture(self.path, live=self._live(), now=fx.NOW)
        r = hp.capture(self.path, live=self._live(fx.nvml(n=2)), force=True, now=fx.NOW + 9)
        self.assertEqual(r["state"], "neu_erfasst")
        doc, _ = hp.load_profile(self.path)
        self.assertEqual(len(doc["cards"]), 2)
        self.assertEqual(doc["capture"], {"at": fx.NOW + 9, "reason": "Neu erfassen"})

    def test_silent_nvml_shows_the_persisted_profile_and_never_overwrites_it(self):
        hp.capture(self.path, live=self._live(), now=fx.NOW)
        raw = open(self.path, encoding="utf-8").read()
        empty = hp.build(cache_dir=os.path.join(self.d.name, "nirgends"), nvml=([], None, ["pynvml nicht lesbar (x)"]), now=fx.NOW)
        r = hp.capture(self.path, live=empty, now=fx.NOW + 1)
        self.assertEqual(r["state"], "nur_gespeichert")
        self.assertEqual(len(r["show"]["cards"]), 3)
        r = hp.capture(self.path, live=empty, force=True, now=fx.NOW + 2)
        self.assertEqual(r["state"], "no_cards")
        self.assertEqual(len(r["show"]["cards"]), 3)
        self.assertIn("nothing saved", r["error"])
        self.assertEqual(open(self.path, encoding="utf-8").read(), raw)

    def test_no_cards_and_no_file_persists_nothing(self):
        empty = hp.build(cache_dir=self.cache, nvml=([], None, []), now=fx.NOW)
        r = hp.capture(self.path, live=empty, now=fx.NOW)
        self.assertEqual(r["state"], "no_cards")
        self.assertFalse(os.path.exists(self.path))

    def test_a_failed_write_is_a_state_not_an_exception(self):
        blocker = os.path.join(self.d.name, "file")
        open(blocker, "w").close()
        r = hp.capture(os.path.join(blocker, "hardware.json"), live=self._live(), now=fx.NOW)     # the "directory" is a file
        self.assertEqual(r["state"], "nicht_schreibbar")
        self.assertIn("Speichern fehlgeschlagen", r["error"])
        self.assertEqual(len(r["show"]["cards"]), 3)

    def test_a_broken_file_is_replaced_at_the_first_call_and_says_so(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as fh:
            fh.write("{kaputt")
        r = hp.capture(self.path, live=self._live(), now=fx.NOW)
        self.assertEqual(r["state"], "erst_erfasst")
        self.assertIn("not readable", r["error"])
        self.assertIsNone(hp.load_profile(self.path)[1])

    def test_another_schema_is_not_taken_for_ours(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as fh:
            json.dump({"schema": "andere/1", "cards": []}, fh)
        doc, problem = hp.load_profile(self.path)
        self.assertIsNone(doc)
        self.assertIn("no flliper.hardware/1", problem)

    def test_the_path_comes_from_the_environment(self):
        self.assertEqual(hp.persist_path({}), "/var/lib/flliper/hardware.json")
        self.assertEqual(hp.persist_path({hp.PERSIST_ENV: "/data/hw.json"}), "/data/hw.json")


class TestLegacyVocabulary(CustomTestCase):
    """F0-D fix round 1: a profile persisted BEFORE the rename spells the source "Datenblatt" (and "borrowed-unbelegt"); the renamed
    module checks "Datasheet".  The file is read in either spelling and is never rewritten by the reader."""

    LIVE = "/var/lib/flliper/hardware.json"

    def _legacy(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        doc = hp.build(cache_dir=d.name, nvml=fx.nvml(), now=fx.NOW)        # no probe file: the SM counts are the data sheet's
        text = json.dumps(doc)
        assert hp.SRC_DATASHEET in text
        return json.loads(text.replace(hp.SRC_DATASHEET, "Datenblatt")), d

    def test_validate_names_the_legacy_source_not_as_unknown(self):
        doc = {"cards": [{"sm_count": {"v": 68, "src": "Datenblatt", "unit": "SM"}}], "links": []}
        self.assertEqual(hp.validate(doc), [])
        doc["cards"][0]["sm_count"]["src"] = "Datenblat"
        self.assertTrue(any("unknown source" in x for x in hp.validate(doc)))

    def test_load_profile_returns_the_translated_vocabulary_and_leaves_the_file_alone(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "hw.json")
            doc = {"schema": hp.SCHEMA, "cards": [{"mem_gbs": {"nominal": {"v": 760, "src": "Datenblatt"}},
                                                    "catalog": {"origin_fields": {"pcie": "Datenblatt", "mem_bw": "borrowed-unbelegt"}}}],
                   "src_vocab": ["gemessen", "NVML", "Datenblatt", "geschätzt", "nicht gemessen"]}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            before = open(p, "rb").read()
            got, problem = hp.load_profile(p)
            self.assertIsNone(problem)
            self.assertEqual(got["cards"][0]["mem_gbs"]["nominal"]["src"], hp.SRC_DATASHEET)
            self.assertEqual(got["cards"][0]["catalog"]["origin_fields"], {"pcie": "Datasheet", "mem_bw": "borrowed-unverified"})
            self.assertEqual(got["src_vocab"][2], hp.SRC_DATASHEET)
            self.assertEqual(open(p, "rb").read(), before)
            self.assertEqual(hp.validate(got), [])

    def test_prose_is_never_translated(self):
        got = hp.normalize_profile({"note": "Datenblatt value of the catalog", "x": ["Datenblatt"]})
        self.assertEqual(got, {"note": "Datenblatt value of the catalog", "x": ["Datasheet"]})

    def test_capture_of_a_legacy_file_is_vorhanden_and_valid(self):
        legacy, d = self._legacy()
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "hw.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(legacy, fh)
            r = hp.capture(p, live=hp.build(cache_dir=d.name, nvml=fx.nvml(), now=fx.NOW), now=fx.NOW)
            self.assertEqual(r["state"], "vorhanden")
            self.assertEqual(hp.validate(r["persisted"]), [])

    @unittest.skipUnless(os.path.exists(LIVE), "the rig's persisted hardware.json is not on this box")
    def test_the_rigs_persisted_file_validates(self):
        got, problem = hp.load_profile(self.LIVE)
        self.assertIsNone(problem)
        self.assertEqual(hp.validate(got), [])
        with open(self.LIVE, encoding="utf-8") as fh:
            self.assertEqual(hp.validate(json.load(fh)), [])        # the raw document too


if __name__ == "__main__":
    unittest.main()
